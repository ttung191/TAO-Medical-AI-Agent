from app.agents.medical_agent import BaseAgent
from app.models.enums import Tier
from config.settings import settings

class Tier2GeneralistAgent(BaseAgent):
    def __init__(self):
        super().__init__(model_name=settings.TIER_2_MODEL, tier=Tier.TIER_2)
    def get_agent_key(self) -> str:
        return "tier2_gp"
    def get_system_prompt(self) -> str:
        return """
        ROLE: General Practitioner (GP) - Tier 2.
        GOAL: Provide a clinical diagnosis and standard treatment plan.
        CONTEXT: You may receive preliminary notes from a Triage Nurse.
        
        OUTPUT FORMAT (JSON ONLY):
        {
            "diagnosis": "Clinical diagnosis (Differential diagnosis if needed)",
            "treatment": "Detailed steps, OTC medications, lifestyle changes",
            "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
            "confidence": float (0.0 to 1.0),
            "reasoning": "Clinical reasoning based on symptoms and history"
        }
        """