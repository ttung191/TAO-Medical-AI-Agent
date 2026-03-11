import logging
from app.core.llm_client import LLMClient
from app.models.schemas import AgentDiagnosis, CostMetrics
from app.models.enums import AgentStatus, EscalationDecision

logger = logging.getLogger(__name__)

class DynamicMedicalAgent:
    def __init__(self, model_name, tier, role_name):
        self.model_name = model_name
        self.tier = tier
        self.role_name = role_name
        self.llm_client = LLMClient()

    def process_case(self, case, history=None, feedback_received=None) -> AgentDiagnosis:
        prompt = f"Patient Symptoms: {case.symptoms}\nHistory: {case.medical_history}\nPrevious Context: {history}\nFeedback: {feedback_received}"
        
        system_prompt = f"You are a {self.role_name} at {self.tier.value}. Analyze and provide diagnosis in JSON format."

        # Sửa lỗi Unpack: Không dùng diag, decision = ...
        res = self.llm_client.generate_json(prompt=prompt, system_prompt=system_prompt, model=self.model_name)

        return AgentDiagnosis(
            agent_name=self.role_name,
            tier=self.tier,
            diagnosis_summary=res.get("diagnosis_summary") or res.get("diagnosis") or "No summary",
            treatment_plan=res.get("treatment_plan") or "No plan",
            confidence_score=float(res.get("confidence_score", 0.8)),
            reasoning=res.get("reasoning") or "No reasoning provided",
            escalation_decision=EscalationDecision(res.get("escalation_decision", "CONTINUE")),
            feedback_to_lower_tier=res.get("feedback_to_lower_tier"),
            metrics=CostMetrics(input_tokens=0, output_tokens=0, total_cost_usd=0.0)
        )
