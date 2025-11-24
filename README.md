# 🌿 Day 3 – Health & Wellness Voice Companion  
### Part of the **Murf AI Voice Agent Challenge**

This repository contains my implementation for **Day 3**, where I built a supportive Health & Wellness companion using **Murf Falcon – the fastest TTS API**.

The agent performs daily voice-based check-ins, reflects on mood and goals, and stores each session in a JSON file for future reference.

---

## 🎯 Primary Goal – Features Implemented

### ✔ 1. Grounded wellness persona  
The agent speaks gently, supportively, and avoids medical claims.  
It behaves like a daily check-in buddy—not a clinician.

### ✔ 2. Voice-based check-ins  
During each session, the agent asks about:
- Mood  
- Energy  
- Stress  
- 1–3 daily goals or intentions  

### ✔ 3. JSON persistence (`wellness_log.json`)  
After each session, the backend stores:
- Timestamp  
- Mood description  
- Goals  
- Short summary sentence  

Example structure:

```json
{
  "sessions": [
    {
      "timestamp": "2025-11-22T10:45:00",
      "mood": "Feeling okay, a bit tired",
      "goals": ["Finish assignment", "Do a 20-min stretch"],
      "summary": "You’re feeling a bit low-energy today and want to stay productive but balanced."
    }
  ]
}