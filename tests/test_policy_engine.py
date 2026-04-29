import pytest

from tao_medical_ai.domain.models import (
    Disposition,
    DomainName,
    FinalDecisionResponse,
    RiskLevel,
)
from tao_medical_ai.safety.policy_engine import PolicyEngine, PolicyViolation


def test_policy_blocks_critical_without_review() -> None:
    engine = PolicyEngine()
    decision = FinalDecisionResponse(
        case_id="bad-001",
        primary_domain=DomainName.cardiology,
        final_risk=RiskLevel.critical,
        final_disposition=Disposition.emergency_department,
        requires_human_review=False,
        rationale="bad",
        clinician_summary="",
        patient_message="",
        evidence=[],
        tier_path=[],
        agent_outputs=[],
        audit_id="a1",
        governance_flags=[],
        complexity_score=9,
    )
    with pytest.raises(PolicyViolation):
        engine.validate(decision)
