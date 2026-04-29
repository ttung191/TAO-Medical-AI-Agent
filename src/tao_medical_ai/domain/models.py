from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"
    critical = "critical"


class EscalationDecision(str, Enum):
    stop = "stop"
    escalate = "escalate"
    review = "review"


class DomainName(str, Enum):
    cardiology = "cardiology"
    neurology = "neurology"
    general_outpatient = "general_outpatient"


class Disposition(str, Enum):
    self_care = "self_care"
    primary_care = "primary_care"
    urgent_care = "urgent_care"
    emergency_department = "emergency_department"
    specialist_review = "specialist_review"
    human_review = "human_review"


class PatientPayload(BaseModel):
    age: int | None = None
    sex: Literal["male", "female", "other"] | None = None


class VitalsPayload(BaseModel):
    spo2: int | None = None
    heart_rate: int | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    respiratory_rate: int | None = None
    temperature_c: float | None = None


class CaseRequest(BaseModel):
    case_id: str = Field(..., min_length=1)
    patient: PatientPayload = Field(default_factory=PatientPayload)
    preferred_domain: DomainName | None = None
    patient_location: str | None = None
    chief_complaint: str = Field(..., min_length=3)
    timeline: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    vitals: VitalsPayload = Field(default_factory=VitalsPayload)
    history: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("symptoms", "history", "medications", "allergies")
    @classmethod
    def clean_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class EvidenceItem(BaseModel):
    evidence_id: str
    title: str
    summary: str
    source: str
    pathway_id: str


class AgentAssessment(BaseModel):
    tier: int
    role: str
    domain: DomainName
    risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    escalation: EscalationDecision
    red_flags: list[str] = Field(default_factory=list)
    missing_critical_data: list[str] = Field(default_factory=list)
    differential: list[str] = Field(default_factory=list)
    rationale: str
    suggested_disposition: Disposition
    handoff_questions: list[str] = Field(default_factory=list)
    structured_findings: dict[str, Any] = Field(default_factory=dict)


class SpecialtyDescriptor(BaseModel):
    slug: DomainName
    label: str
    description: str
    intended_use: str
    required_intake_fields: list[str] = Field(default_factory=list)
    default_handoff_questions: list[str] = Field(default_factory=list)


class FinalDecisionResponse(BaseModel):
    case_id: str
    primary_domain: DomainName
    service_line: str = ""
    final_risk: RiskLevel
    final_disposition: Disposition
    requires_human_review: bool
    human_review_reason: str | None = None
    rationale: str
    clinician_summary: str
    patient_message: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tier_path: list[str] = Field(default_factory=list)
    agent_outputs: list[AgentAssessment] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    intake_gaps: list[str] = Field(default_factory=list)
    audit_id: str
    governance_flags: list[str] = Field(default_factory=list)
    complexity_score: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReviewRecord(BaseModel):
    audit_id: str
    case_id: str
    risk: RiskLevel
    disposition: Disposition
    reason: str


class ReviewDecision(BaseModel):
    approved: bool
    reviewer: str
    note: str | None = None
