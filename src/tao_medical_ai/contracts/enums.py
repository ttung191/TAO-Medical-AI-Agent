from enum import Enum

class RiskLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"
    critical = "critical"

class EscalationDecision(str, Enum):
    stop = "stop"
    review = "review"
    escalate = "escalate"

class Disposition(str, Enum):
    self_care = "self_care"
    primary_care = "primary_care"
    urgent_care = "urgent_care"
    emergency_department = "emergency_department"
    human_review = "human_review"

class DomainName(str, Enum):
    cardiology = "cardiology"
    neurology = "neurology"
    general_outpatient = "general_outpatient"