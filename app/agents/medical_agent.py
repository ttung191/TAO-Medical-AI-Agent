import logging
from abc import ABC, abstractmethod
from app.core.llm_client import LLMClient
from app.models.schemas import AgentDiagnosis, CostMetrics
from app.models.enums import AgentStatus, EscalationDecision, RiskLevel

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
            "Analyze and provide diagnosis in JSON format. "
            "Fields: diagnosis_summary, treatment_plan, confidence_score, reasoning, "
            "risk_assessment (LOW/MODERATE/HIGH/CRITICAL), "
            "escalation_decision (ESCALATE/REJECT/COMPLETED)."
        )

        try:
            res = self.llm_client.generate_json(prompt=prompt, system_prompt=system_prompt, model=self.model_name)
            if not isinstance(res, dict): res = {}

            # ✅ 1. Xử lý an toàn cho EscalationDecision
            raw_decision = str(res.get("escalation_decision", "ESCALATE")).upper().strip()
            if raw_decision not in ["ESCALATE", "REJECT", "COMPLETED"]:
                final_decision = EscalationDecision.ESCALATE
            else:
                final_decision = EscalationDecision(raw_decision)

            # ✅ 2. Xử lý an toàn cho RiskLevel (Fix triệt để lỗi MEDIUM)
            raw_risk = str(res.get("risk_assessment", "HIGH")).upper().strip()
            # Tự động nắn chỉnh nếu AI trả về MEDIUM thay vì MODERATE
            if raw_risk == "MEDIUM": 
                raw_risk = "MODERATE"
                
            try:
                final_risk = RiskLevel(raw_risk)
            except ValueError:
                # Nếu AI trả về từ vựng quá kỳ lạ, mặc định gán là HIGH cho an toàn y khoa
                final_risk = RiskLevel.HIGH 

            return AgentDiagnosis(
                agent_name=self.role_name,
                role=self.role_name, 
                tier=self.tier,
                diagnosis_summary=res.get("diagnosis_summary") or res.get("diagnosis") or "No summary",
                treatment_plan=res.get("treatment_plan") or "No plan",
                confidence_score=float(res.get("confidence_score", 0.8)),
                risk_assessment=final_risk, # Đã dùng biến an toàn
                reasoning=res.get("reasoning") or "No reasoning provided",
                escalation_decision=final_decision,
                feedback_to_lower_tier=res.get("feedback_to_lower_tier"),
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
                role=self.role_name,
                tier=self.tier,
                diagnosis_summary="Lỗi xử lý",
                treatment_plan="Cần khởi động lại",
                confidence_score=0.0,
                risk_assessment=RiskLevel.HIGH, # Dùng cứng mức HIGH khi có lỗi
                reasoning=str(e),
                escalation_decision=EscalationDecision.REJECT,
                metrics=CostMetrics(model_name=self.model_name, input_tokens=0, output_tokens=0, total_cost_usd=0.0)
            )
