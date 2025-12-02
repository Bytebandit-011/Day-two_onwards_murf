import logging
import json
import random
from datetime import datetime
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Improv scenarios
IMPROV_SCENARIOS = [
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension.",
    "You are a time-travelling tour guide explaining modern smartphones to someone from the 1800s.",
    "You are a restaurant waiter who must calmly tell a customer that their order has escaped the kitchen.",
    "You are a customer trying to return an obviously cursed object to a very skeptical shop owner.",
    "You are a therapist conducting a session with a superhero who's afraid of heights.",
    "You are a museum guide explaining to visitors why all the paintings have mysteriously turned into memes.",
    "You are a driving instructor teaching someone who thinks they're in a video game.",
    "You are a job interviewer for a position as a professional napper.",
    "You are a tech support agent helping someone whose toaster has gained sentience.",
    "You are a real estate agent showing a house that's clearly haunted, but you're in denial about it.",
    "You are a flight attendant announcing that the plane has accidentally time-traveled to the dinosaur era.",
    "You are a librarian scolding someone for being too quiet.",
    "You are a chef on a cooking show where every ingredient is invisible.",
    "You are a bank teller processing a withdrawal for a pirate who insists on paying in doubloons.",
    "You are a personal trainer coaching a zombie who wants to get in shape.",
]


class ImprovGameState:
    """Manages the state of the improv game"""
    
    def __init__(self):
        self.player_name: Optional[str] = None
        self.current_round: int = 0
        self.max_rounds: int = 3
        self.rounds: list[dict] = []
        self.phase: str = "intro"  # "intro" | "awaiting_improv" | "reacting" | "done"
        self.used_scenarios: list[str] = []
        self.current_scenario: Optional[str] = None
        self.turn_count_in_scene: int = 0
        self.scene_ended: bool = False
    
    def get_next_scenario(self) -> str:
        """Get a random scenario that hasn't been used yet"""
        available = [s for s in IMPROV_SCENARIOS if s not in self.used_scenarios]
        if not available:
            # If all used, reset
            self.used_scenarios = []
            available = IMPROV_SCENARIOS
        
        scenario = random.choice(available)
        self.used_scenarios.append(scenario)
        self.current_scenario = scenario
        return scenario
    
    def start_new_round(self):
        """Initialize a new round"""
        self.current_round += 1
        self.phase = "awaiting_improv"
        self.turn_count_in_scene = 0
        self.scene_ended = False
        scenario = self.get_next_scenario()
        self.rounds.append({
            "round_number": self.current_round,
            "scenario": scenario,
            "player_performance": [],
            "host_reaction": None
        })
        return scenario
    
    def add_player_line(self, text: str):
        """Record a player's improv line"""
        if self.rounds:
            self.rounds[-1]["player_performance"].append(text)
            self.turn_count_in_scene += 1
    
    def end_current_round(self, host_reaction: str):
        """End the current round with host's reaction"""
        if self.rounds:
            self.rounds[-1]["host_reaction"] = host_reaction
        self.phase = "reacting"
    
    def is_game_complete(self) -> bool:
        """Check if all rounds are done"""
        return self.current_round >= self.max_rounds
    
    def should_end_scene(self) -> bool:
        """Determine if the current scene should end"""
        # End after 3-5 player turns or if explicitly ended
        return self.scene_ended or self.turn_count_in_scene >= random.randint(3, 5)


class Assistant(Agent):
    def __init__(self, game_state: ImprovGameState) -> None:
        super().__init__(
            instructions="""You are the charismatic host of a TV improv show called "Improv Battle"!

Your personality:
- High-energy, witty, and enthusiastic about improv
- Clear and direct when explaining rules and scenarios
- Your reactions are VARIED and REALISTIC - not always supportive
- Sometimes you're amused, sometimes unimpressed, sometimes pleasantly surprised
- You give honest critiques but stay constructive and respectful
- You can lightly tease but never be cruel or abusive
- You celebrate creativity and bold choices

Your role in the game:
1. Introduce the show warmly and explain how it works
2. Present each improv scenario clearly to the player
3. Let the player perform their improv
4. React authentically to their performance with specific observations
5. Move the game forward between rounds
6. Provide a final summary at the end

When reacting to performances:
- Mix your tones: sometimes supportive ("That was hilarious!"), sometimes constructive ("You could have pushed that further"), sometimes mildly critical ("That felt a bit safe")
- Be SPECIFIC about what you noticed: reference actual choices they made
- Vary your energy: don't use the same reaction pattern every time
- Stay encouraging overall, but don't fake enthusiasm
- Comment on their character work, timing, commitment, creativity, or absurdity

Keep your responses conversational and concise - this is a voice interaction. Avoid overly long explanations.""",
        )
        self.game_state = game_state
        self.last_reaction_tone = None  # Track to vary reactions
    
    @function_tool
    async def set_player_name(self, context: RunContext, name: str):
        """Set the player's name at the start of the game.
        
        Args:
            name: The player's name
        """
        logger.info(f"Setting player name: {name}")
        self.game_state.player_name = name
        return f"Great! Welcome to Improv Battle, {name}! Let me explain how this works."
    
    @function_tool
    async def start_next_round(self, context: RunContext):
        """Start the next improv round with a new scenario.
        
        Use this when you're ready to present the next improv challenge to the player.
        """
        logger.info(f"Starting round {self.game_state.current_round + 1}")
        
        if self.game_state.is_game_complete():
            self.game_state.phase = "done"
            return "All rounds complete! Time for the final summary."
        
        scenario = self.game_state.start_new_round()
        
        return f"Round {self.game_state.current_round} of {self.game_state.max_rounds}. Here's your scenario: {scenario}. Begin your improv whenever you're ready!"
    
    @function_tool
    async def end_current_scene(self, context: RunContext, reaction: str):
        """End the current improv scene and provide your host reaction.
        
        Use this after the player has performed their improv and you want to give feedback.
        The reaction should be specific, varied in tone, and authentic.
        
        Args:
            reaction: Your detailed reaction to the player's performance
        """
        logger.info(f"Ending scene with reaction")
        
        self.game_state.end_current_round(reaction)
        
        # Track reaction tone to encourage variety
        if "hilarious" in reaction.lower() or "loved" in reaction.lower():
            self.last_reaction_tone = "positive"
        elif "flat" in reaction.lower() or "could have" in reaction.lower():
            self.last_reaction_tone = "critical"
        else:
            self.last_reaction_tone = "neutral"
        
        return f"Reaction recorded. Moving forward in the game."
    
    @function_tool
    async def get_game_status(self, context: RunContext):
        """Get the current status of the game.
        
        Use this to check where we are in the game flow.
        """
        status = {
            "player_name": self.game_state.player_name,
            "current_round": self.game_state.current_round,
            "max_rounds": self.game_state.max_rounds,
            "phase": self.game_state.phase,
            "current_scenario": self.game_state.current_scenario,
            "is_complete": self.game_state.is_game_complete()
        }
        
        logger.info(f"Game status: {status}")
        return json.dumps(status, indent=2)
    
    @function_tool
    async def end_game_early(self, context: RunContext):
        """End the game early if the player wants to stop.
        
        Use this if the player clearly indicates they want to quit or stop playing.
        """
        logger.info("Game ended early by player request")
        self.game_state.phase = "done"
        return "Game ended. Prepare a brief farewell for the player."
    
    @function_tool
    async def mark_scene_complete(self, context: RunContext):
        """Mark that the player has indicated their scene is complete.
        
        Use this when the player says something like 'end scene', 'that's it', or clearly signals they're done.
        """
        logger.info("Player marked scene as complete")
        self.game_state.scene_ended = True
        return "Scene marked as complete. Provide your reaction now."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Initialize game state for this session
    game_state = ImprovGameState()
    logger.info("New Improv Battle game session started")

    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Track player turns for auto-scene-ending
    @session.on("user_speech_committed")
    def on_user_speech(msg):
        """Track player's improv lines"""
        if game_state.phase == "awaiting_improv":
            text = msg.text if hasattr(msg, 'text') else str(msg)
            game_state.add_player_line(text)
            logger.info(f"Player turn {game_state.turn_count_in_scene}: {text[:50]}...")
            
            # Check if scene should auto-end
            if game_state.should_end_scene():
                logger.info(f"Scene auto-ending after {game_state.turn_count_in_scene} turns")

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        logger.info(f"Game completed: {game_state.is_game_complete()}, Rounds played: {game_state.current_round}")
    
    async def save_session():
        """Save the game session to JSON file"""
        sessions_dir = Path(__file__).parent.parent / "game_sessions"
        sessions_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = sessions_dir / f"SESSION_{timestamp}.json"
        
        session_data = {
            "player_name": game_state.player_name,
            "current_round": game_state.current_round,
            "max_rounds": game_state.max_rounds,
            "rounds": game_state.rounds,
            "phase": game_state.phase,
            "timestamp": timestamp,
            "completed": game_state.is_game_complete()
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session saved to {filename}")

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(save_session)

    # Start the session
    await session.start(
        agent=Assistant(game_state),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()
    
    logger.info("Agent connected and ready for Improv Battle!")



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))