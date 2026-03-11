import logging
from typing import Dict, List
from app.core.llm_client import LLMClient
from app.models.enums import Tier

logger = logging.getLogger(__name__)

class AgentRecruiter:
    def __init__(self):
        self.llm_client = LLMClient()

    def recruit_and_route(self, case_summary: str) -> Dict[Tier, List[str]]:
        prompt = f"Based on this case: {case_summary}, recruit specialized medical agents for Tier 1, 2, and 3."
        system_prompt = "Return JSON: {'TIER_1': ['Role1'], 'TIER_2': ['Role2'], 'TIER_3': ['Role3']}"
        
        try:
            response = self.llm_client.generate_json(prompt=prompt, system_prompt=system_prompt)
            recruitment = {}
            if isinstance(response, dict):
                for tier_str, roles in response.items():
                    try:
                        key = Tier[tier_str.upper()]
                        recruitment[key] = roles if isinstance(roles, list) else [str(roles)]
                    except: continue
            return recruitment if recruitment else self._default_team()
        except Exception as e:
            logger.error(f"⚠️ Recruitment failed: {e}")
            return self._default_team()

    def _default_team(self):
        return {
            Tier.TIER_1: ["General_Nurse"],
            Tier.TIER_2: ["General_Practitioner"],
            Tier.TIER_3: ["Senior_Consultant"]
        }
