import json
from app.core.llm_client import LLMClient
from app.models.enums import Tier

class AgentRecruiter:
    def __init__(self):
        self.llm = LLMClient()

    def recruit_and_route(self, case_text: str) -> dict:
        """
        Phân tích ca bệnh và quyết định đội ngũ y tế cần thiết (Dynamic Recruitment)
        """
        system_prompt = """
        You are a Medical Chief of Staff. Analyze the patient case and recruit a team of AI medical agents.
        
        RULES:
        1. Identify 2-3 specific roles needed (e.g., Cardiologist, Toxicologist, Nurse).
        2. Assign them to Tiers based on complexity:
           - TIER_1: Initial Assessment (Generalists, Nurses).
           - TIER_2: Specialists (Cardiologists, Neurologists).
           - TIER_3: Super-Specialists/Consultants (Ethics Board, Rare Disease Expert).
        3. Ensure at least one agent in Tier 1.
        4. ALWAYS assign a "Clinical_Supervisor" or "Chief_Medical_Officer" to TIER_3_CONSULTANT if no specific super-specialist is obvious. TIER 3 MUST NOT BE EMPTY.
        
        OUTPUT JSON FORMAT:
        {
            "TIER_1_INITIAL": ["Role Name 1"],
            "TIER_2_SPECIALIST": ["Role Name 2"],
            "TIER_3_CONSULTANT": ["Role Name 3"]
        }
        """
        
        # Dùng model mặc định (Flash) được cấu hình trong LLMClient
        response, _ = self.llm.generate_json(
            model="models/gemini-2.5-flash", 
            system_prompt=system_prompt,
            user_prompt=f"PATIENT CASE: {case_text}"
        )
        
        # Validate và map về Enum
        recruitment_plan = {
            Tier.TIER_1: response.get("TIER_1_INITIAL", ["General_Nurse"]),
            Tier.TIER_2: response.get("TIER_2_SPECIALIST", ["General_Practitioner"]),
            Tier.TIER_3: response.get("TIER_3_CONSULTANT", ["Clinical_Supervisor"]) # Fallback an toàn
        }
        
        # Double check: Nếu Tier 3 rỗng, tự thêm vào
        if not recruitment_plan[Tier.TIER_3]:
            recruitment_plan[Tier.TIER_3] = ["Clinical_Supervisor"]
            
        return recruitment_plan