from app.agents.medical_agent import BaseAgent
from app.models.enums import Tier
from config.settings import settings

class Tier1NurseAgent(BaseAgent):
    def __init__(self):
        super().__init__(model_name=settings.TIER_1_MODEL, tier=Tier.TIER_1)
    def get_agent_key(self) -> str:
        return "tier1_nurse"

    def get_system_prompt(self) -> str:
        return """
        ROLE: Triage Nurse (Medical AI Tier 1).
        GOAL: Assess patient symptoms quickly, identify urgency, and suggest basic home care.
        CONSTRAINTS: 
        - DO NOT prescribe prescription drugs.
        - Be conservative. If unsure, mark confidence low.
        
        OUTPUT FORMAT (JSON ONLY):
        {
            "diagnosis": "Brief possible condition",
            "treatment": "Immediate advice (e.g., rest, hydration, go to ER)",
            "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
            "confidence": float (0.0 to 1.0),
            "reasoning": "Why you chose this risk level"
        }
        """