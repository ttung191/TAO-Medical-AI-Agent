from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from app.models.enums import Tier, RiskLevel, AgentStatus, EscalationDecision

class CostMetrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0
    model_name: str
    model_config = {'protected_namespaces': ()}

class PatientCase(BaseModel):
    case_id: str
    symptoms: str
    medical_history: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentDiagnosis(BaseModel):
    tier: Tier
    agent_name: str  # Ví dụ: "Cardiologist_Agent"
    role: str        # Ví dụ: "Cardiologist"
    diagnosis_summary: str
    treatment_plan: str
    confidence_score: float
    risk_assessment: RiskLevel
    reasoning: str
    
    # --- MỚI: Các trường hỗ trợ luồng phức tạp ---
    escalation_decision: EscalationDecision = EscalationDecision.RESOLVE
    feedback_to_lower_tier: Optional[str] = None # Feedback khi trả hồ sơ
    
    metrics: CostMetrics

class SystemState(BaseModel):
    case: PatientCase
    current_tier: Tier
    status: AgentStatus
    
    # Danh sách các agent được tuyển dụng (Dynamic)
    recruited_agents: dict = {} # {Tier.TIER_1: ["Nurse", "GP"], ...}
    
    # Lịch sử hội thoại (List để lưu nhiều vòng lặp Reject/Escalate)
    interaction_history: List[AgentDiagnosis] = []
    
    final_diagnosis: Optional[AgentDiagnosis] = None
    
    total_cost: float = 0.0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    logs: List[str] = [] # Lưu log chi tiết quá trình