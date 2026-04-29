from tao_medical_ai.app.bootstrap import build_orchestrator
from tao_medical_ai.domain.models import CaseRequest, DomainName, PatientPayload, VitalsPayload


def test_high_risk_cardiology_case_requires_review() -> None:
    orchestrator = build_orchestrator()
    payload = CaseRequest(
        case_id="cardio-001",
        preferred_domain=DomainName.cardiology,
        patient=PatientPayload(age=72, sex="male"),
        chief_complaint="Chest pain and shortness of breath",
        timeline="Started 1 hour ago with diaphoresis while walking",
        symptoms=["chest pain", "dyspnea", "diaphoresis"],
        vitals=VitalsPayload(spo2=88, heart_rate=128, systolic_bp=86),
        history=["hypertension"],
        notes="Pressure sensation radiating to left arm.",
    )
    result = orchestrator.run(payload)
    assert result.primary_domain.value == "cardiology"
    assert result.requires_human_review is True
    assert result.final_disposition.value == "human_review"
    assert result.final_risk.value == "critical"
    assert "tier3_specialist" in result.tier_path


def test_stroke_case_routes_to_neurology() -> None:
    orchestrator = build_orchestrator()
    payload = CaseRequest(
        case_id="neuro-001",
        patient=PatientPayload(age=68, sex="female"),
        chief_complaint="Sudden slurred speech and right-sided weakness",
        timeline="Started 25 minutes ago, last known well 10 minutes before onset",
        symptoms=["slurred speech", "one-sided weakness", "facial droop"],
        vitals=VitalsPayload(spo2=97, heart_rate=90, systolic_bp=170),
        history=["atrial fibrillation"],
    )
    result = orchestrator.run(payload)
    assert result.primary_domain.value == "neurology"
    assert result.final_risk.value == "critical"
    assert result.requires_human_review is True


def test_low_risk_general_outpatient_can_avoid_specialist() -> None:
    orchestrator = build_orchestrator()
    payload = CaseRequest(
        case_id="outpatient-001",
        preferred_domain=DomainName.general_outpatient,
        patient=PatientPayload(age=21, sex="female"),
        chief_complaint="Mild sore throat",
        timeline="Since yesterday",
        symptoms=["sore throat"],
        vitals=VitalsPayload(spo2=99, heart_rate=80, systolic_bp=118, temperature_c=37.3),
        notes="Drinking fluids normally.",
    )
    result = orchestrator.run(payload)
    assert result.primary_domain.value == "general_outpatient"
    assert result.final_risk.value in {"low", "moderate"}
    assert result.audit_id
    assert "tier3_specialist" not in result.tier_path
