import logging
from abc import ABC, abstractmethod
from app.core.llm_client import LLMClient
from app.models.schemas import AgentDiagnosis, CostMetrics
from app.models.enums import AgentStatus, EscalationDecision

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, model_name, tier, role_name):
        self.model_name = model_name
        self.tier = tier
        self.role_name = role_name
        self.llm_client = LLMClient()

    @abstractmethod
    def process_case(self, case, history=None, feedback_received=None):
        pass

class DynamicMedicalAgent(BaseAgent):
    def process_case(self, case, history=None, feedback_received=None) -> AgentDiagnosis:
        prompt = (
            f"Patient Symptoms: {case.symptoms}\n"
            f"History: {case.medical_history}\n"
            f"Previous Context: {history}\n"
            f"Feedback: {feedback_received}"
        )
        
        system_prompt = (
            f"You are a {self.role_name} at {self.tier.value}. "
            "Analyze the case and provide a diagnosis in JSON format. "
            "Fields: diagnosis_summary, treatment_plan, confidence_score, "
            "risk_assessment, reasoning, escalation_decision, feedback_to_lower_tier."
        )

        try:
            res = self.llm_client.generate_json(prompt=prompt, system_prompt=system_prompt, model=self.model_name)
            
            if not isinstance(res, dict):
                res = {}

            return AgentDiagnosis(
                agent_name=self.role_name,
                tier=self.tier,
                diagnosis_summary=res.get("diagnosis_summary") or res.get("diagnosis") or "No summary",
                treatment_plan=res.get("treatment_plan") or "No plan",
                confidence_score=float(res.get("confidence_score", 0.8)),
                reasoning=res.get("reasoning") or "No reasoning provided",
                escalation_decision=EscalationDecision(res.get("escalation_decision", "CONTINUE")),
                feedback_to_lower_tier=res.get("feedback_to_lower_tier"),
                # ✅ ĐÃ FIX: Thêm model_name vào đây
                metrics=CostMetrics(
                    model_name=self.model_name, 
                    input_tokens=0, 
                    output_tokens=0, 
                    total_cost_usd=0.0
                )
            )
        except Exception as e:
            logger.error(f"❌ Agent {self.role_name} Error: {e}")
            return AgentDiagnosis(
                agent_name=self.role_name,
                tier=self.tier,
                diagnosis_summary="Lỗi xử lý Agent",
                treatment_plan="Cần hội chẩn lại",
                confidence_score=0.0,
                reasoning=str(e),
                escalation_decision=EscalationDecision.REJECT,
                metrics=CostMetrics(
                    model_name=self.model_name, 
                    input_tokens=0, 
                    output_tokens=0, 
                    total_cost_usd=0.0
                )
            )
