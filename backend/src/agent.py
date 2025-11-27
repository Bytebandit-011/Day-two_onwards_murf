import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("fraud-agent")
load_dotenv(".env.local")

class FraudAgent(Agent):
    def __init__(self) -> None:
        self.current_case = None
        self.verified = False 

        super().__init__(
            instructions = """You are a fraud detection representative agent for HDFC bank.

YOUR ROLE:
- You are calling customers about suspicious transactions that were flagged
- Be professional, calm, and reassuring
- NEVER ask for full card numbers, PINs, or passwords

CALL FLOW (FOLLOW EXACTLY):
1. Greet: "Hello, this is the fraud prevention department from HDFC Bank."
2. Ask: "To verify if any fraudlent transcation might have been done in your account or not may i have your name?"
3. Get their name and use load_fraud_case tool
4. If no case found, apologize and end call
5. If case found, ask Before we continue, can you answer a security question using verify_customer tool to ask the question
6. If verification FAILS:
   - Say: "I'm sorry, that doesn't match our records. Please call the number on your card."
   - Use update_case_status with "verification_failed"
   - End call
7. If verification PASSES:
   - Read transaction details clearly
   - Ask: "Did you make this purchase?"
8. Based on their answer:
   - YES = use update_case_status with "confirmed_safe"
   - NO = use update_case_status with "confirmed_fraud" and tell them card is blocked
9. Thank them and end call

IMPORTANT:
- Use tools immediately when needed
- Be concise and clear
- No bullet points or formatting in speech""",
            
        )
    
    @function_tool
    async def load_fraud_case(self, context: RunContext, user_name: str):
        """Load pending fraud case for customer.
        
        Call this after customer confirms their name.
        
        Args:
            user_name: Customer's full name
            
        Returns:
            Case details and security question if found
        """
        cases_file = Path("DB/fraud_cases.json")
        
        with open(cases_file, "r") as f:
            cases = json.load(f)

        # Find pending case for this user
        for case in cases:
            if case['userName'].lower() == user_name.lower() and case['status'] == 'pending_review':
                self.current_case = case
                logger.info(f"Loaded case {case['id']} for {user_name}")
                return f"Case {case['id']} loaded. Security question: {case['securityQuestion']}"

        logger.warning(f"No pending cases for {user_name}")
        return "No pending fraud cases found for this customer"

    @function_tool
    async def verify_customer(self, context: RunContext, user_answer: str):
        """Verify customer identity with security question answer."""
        if not self.current_case:
            return "No case loaded"
        
        expected = self.current_case['securityAnswer'].lower().strip()
        provided = user_answer.lower().strip()

        if expected == provided:
            self.verified = True
            logger.info(f"Customer verified for case {self.current_case['id']}")
            
            # Return transaction details with EXACT field names
            return f"""Verified! Transaction details:
- Amount: ${self.current_case['transactionAmount']} {self.current_case['transactionCurrency']}
- Merchant: {self.current_case['transactionName']}
- Time: {self.current_case['transactionTime']}
- Location: {self.current_case['transactionLocation']}
- Card ending: {self.current_case['cardEnding']}"""
        else:
            self.verified = False
            logger.warning(f"Verification failed for case {self.current_case['id']}")
            return "failed"

    @function_tool
    async def update_case_status(self, context: RunContext, status: str, outcome: str):
        """Update fraud case status in database.
        
        Call this when the verification is complete.
        
        Args:
            status: "confirmed_safe", "confirmed_fraud", or "verification_failed"
            outcome: Description of what happened
            
        Returns:
            Confirmation message
        """
        if not self.current_case:
            return "No case to update"
        
        cases_file = Path("DB/fraud_cases.json")
        
        # Load all cases
        with open(cases_file, "r") as f:
            cases = json.load(f)
        
        # Find and update this case
        for case in cases:
            if case['id'] == self.current_case['id']:
                case['status'] = status
                case['outcome'] = outcome
                case['verificationTimestamp'] = datetime.now().isoformat()
                break
        
        # Save back to file
        with open(cases_file, "w") as f:
            json.dump(cases, f, indent=2)
        
        logger.info(f"✓ Case {self.current_case['id']} updated to: {status}")
        return f"Case updated to {status}"


def prewarm(proc: JobProcess):
    """Pre-load models to reduce latency"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entry point for the fraud alert agent"""
    
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Create voice pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-ryan",  # Professional male voice
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
    )
    
    await ctx.connect()
    
    logger.info("🚨 Fraud Alert Agent ready!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))