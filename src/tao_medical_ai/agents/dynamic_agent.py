import json, logging
from tao_medical_ai.infrastructure.llm.provider import LLMFactory
from tao_medical_ai.contracts.case import StructuredCase, AgentAssessment, TokenUsage
from tao_medical_ai.contracts.enums import RiskLevel, EscalationDecision, Disposition
from tao_medical_ai.security.phi_filter import PHIFilter

logger = logging.getLogger(__name__)

class DynamicMedicalAgent:
    def __init__(self, tier: int, role: str):
        self.tier = tier
        self.role = role
        self.llm_provider = LLMFactory.get_provider(self.tier)

    async def run(self, case: StructuredCase, history: dict) -> AgentAssessment:
        prior_notes = history.get('prior_notes', 'Chưa có')
        prompt = f"""Bạn là {self.role} (Tier {self.tier}). 
        LỊCH SỬ TUYẾN DƯỚI: {prior_notes}
        BỆNH ÁN: {case.to_agent_prompt()}

        Hãy trả về JSON (chỉ JSON) với các key: "working_diagnosis", "risk" ("low"|"moderate"|"high"|"critical"), "escalation" ("stop"|"review"|"escalate"), "suggested_disposition" ("self_care"|"primary_care"|"urgent_care"|"emergency_department"|"human_review"), "rationale".
        """
        
        safe_prompt = PHIFilter.redact(prompt)
        llm_resp = await self.llm_provider.generate_assessment(safe_prompt)
        
        try:
            data = json.loads(llm_resp.text)
        except Exception as e:
            logger.error(f"Lỗi JSON: {e}")
            data = {"risk": "high", "escalation": "escalate", "suggested_disposition": "human_review", "working_diagnosis": "Lỗi", "rationale": "Fallback an toàn do lỗi parse"}

        return AgentAssessment(
            tier=self.tier, role=self.role,
            risk=RiskLevel(data.get("risk", "high")),
            escalation=EscalationDecision(data.get("escalation", "escalate")),
            differential=[data.get("working_diagnosis", "Unknown")],
            rationale=data.get("rationale", ""),
            suggested_disposition=Disposition(data.get("suggested_disposition", "human_review")),
            token_usage=TokenUsage(
                prompt_tokens=llm_resp.prompt_tokens, completion_tokens=llm_resp.completion_tokens,
                total_tokens=llm_resp.prompt_tokens + llm_resp.completion_tokens
            )
        )