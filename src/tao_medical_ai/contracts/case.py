from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from tao_medical_ai.contracts.enums import RiskLevel, EscalationDecision, Disposition, DomainName

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class StructuredCase(BaseModel):
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_age: Optional[int] = Field(None, ge=0, le=120)
    patient_sex: Optional[str] = Field(None)
    chief_complaint: str = Field(..., min_length=5)
    symptoms: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    vitals: Optional[Dict[str, str]] = Field(default_factory=dict)
    labs: Optional[Dict[str, str]] = Field(default_factory=dict)

    def to_agent_prompt(self) -> str:
        prompt = f"- Lý do khám: {self.chief_complaint}\n"
        if self.patient_age and self.patient_sex:
            prompt += f"- Bệnh nhân: {self.patient_sex}, {self.patient_age} tuổi\n"
        prompt += f"- Triệu chứng: {', '.join(self.symptoms)}\n"
        prompt += f"- Tiền sử: {', '.join(self.medical_history)}\n"
        prompt += f"- Sinh hiệu: {self.vitals}\n"
        prompt += f"- Xét nghiệm: {self.labs}\n"
        return prompt

class AgentAssessment(BaseModel):
    tier: int
    role: str
    risk: RiskLevel
    escalation: EscalationDecision
    differential: List[str]
    rationale: str
    suggested_disposition: Disposition
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

class RecruitedAgent(BaseModel):
    expertise: str
    reason: str

class RoutedAgent(BaseModel):
    expertise: str
    tier: int