from app.agents.medical_agent import BaseAgent
from app.models.enums import Tier
from config.settings import settings

class Tier3SpecialistAgent(BaseAgent):
    def __init__(self):
        super().__init__(model_name=settings.TIER_3_MODEL, tier=Tier.TIER_3)
    def get_agent_key(self) -> str:
        return "tier3_specialist"
    def get_system_prompt(self) -> str:
        return """
        ROLE: Senior Medical Specialist Board (Tier 3).
        GOAL: Handle complex, high-risk, or ambiguous cases escalated by lower tiers.
        TASK:
        1. Review the patient case deeply.
        2. Critique previous opinions if they exist.
        3. Provide the most safe and accurate medical advice possible.
        
        OUTPUT FORMAT (JSON ONLY):
        {
            "diagnosis": "Final authoritative diagnosis",
            "treatment": "Comprehensive plan including hospitalization if needed",
            "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
            "confidence": float (0.0 to 1.0),
            "reasoning": "In-depth analysis of why previous tiers might have escalated"
        }
        """