import json
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
        system_prompt = """
        You are a Medical Director. Return a JSON object with keys: 'TIER_1', 'TIER_2', 'TIER_3'.
        Value for each key must be a list of roles (strings).
        Example: {"TIER_1": ["General_Nurse"], "TIER_2": ["Cardiologist"], "TIER_3": ["Senior_Consultant"]}
        """
        
        try:
            # Sửa lỗi Unpack bằng cách nhận 1 biến duy nhất
            response = self.llm_client.generate_json(prompt=prompt, system_prompt=system_prompt)
            
            # Mapping an toàn từ chuỗi sang Enum Tier
            recruitment = {}
            for tier_str, roles in response.items():
                try:
                    tier_enum = Tier[tier_str.upper()]
                    recruitment[tier_enum] = roles if isinstance(roles, list) else [str(roles)]
                except:
                    continue
            
            if not recruitment: raise ValueError("Empty recruitment plan")
            return recruitment
            
        except Exception as e:
            logger.error(f"Recruitment Error: {e}")
            # Trả về team mặc định nếu lỗi
            return {
                Tier.TIER_1: ["General_Nurse"],
                Tier.TIER_2: ["General_Practitioner"],
                Tier.TIER_3: ["Senior_Consultant"]
            }
