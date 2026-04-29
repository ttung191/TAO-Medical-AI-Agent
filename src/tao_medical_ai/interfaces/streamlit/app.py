from __future__ import annotations

import json

import streamlit as st

from tao_medical_ai.app.bootstrap import build_orchestrator
from tao_medical_ai.domain.models import CaseRequest, DomainName, PatientPayload, VitalsPayload
from tao_medical_ai.domain.specialties import list_specialties

PRESETS = {
    DomainName.cardiology: {
        "chief_complaint": "Chest pain and shortness of breath",
        "timeline": "Started 2 hours ago while climbing stairs",
        "symptoms": "chest pain, dyspnea, diaphoresis",
        "history": "hypertension, type 2 diabetes",
        "notes": "Pain feels like pressure and is radiating to left arm.",
        "spo2": 91,
        "hr": 116,
        "sbp": 94,
        "temperature": 37.0,
    },
    DomainName.neurology: {
        "chief_complaint": "Sudden slurred speech and right arm weakness",
        "timeline": "Started 35 minutes ago, last known well 30 minutes before onset",
        "symptoms": "speech difficulty, one-sided weakness, facial droop",
        "history": "atrial fibrillation, hypertension",
        "notes": "Family reports abrupt onset. Possible anticoagulant use.",
        "spo2": 97,
        "hr": 98,
        "sbp": 168,
        "temperature": 36.8,
    },
    DomainName.general_outpatient: {
        "chief_complaint": "Sore throat and mild cough",
        "timeline": "Since yesterday evening",
        "symptoms": "sore throat, cough, fatigue",
        "history": "seasonal allergies",
        "notes": "Tolerating fluids. No chest pain or focal neurologic deficits.",
        "spo2": 99,
        "hr": 82,
        "sbp": 118,
        "temperature": 37.4,
    },
}

st.set_page_config(page_title="TAO Medical AI Agent v3", layout="wide")
st.title("TAO Medical AI Agent v3")
st.caption("Multi-specialty TAO workspace: cardiology triage, neurology stroke triage, and general outpatient assistant")

specialty_options = {descriptor.label: descriptor.slug for descriptor in list_specialties()}
selected_label = st.selectbox("Specialty workflow", options=list(specialty_options))
selected_domain = specialty_options[selected_label]
preset = PRESETS[selected_domain]

with st.expander("Intake checklist", expanded=False):
    descriptor = next(item for item in list_specialties() if item.slug == selected_domain)
    st.write("Required intake fields:")
    st.json(descriptor.required_intake_fields)
    st.write("Default handoff questions:")
    st.json(descriptor.default_handoff_questions)

with st.form("triage-form"):
    col1, col2 = st.columns(2)
    with col1:
        case_id = st.text_input("Case ID", value=f"demo-{selected_domain.value}-001")
        age = st.number_input("Age", min_value=0, max_value=120, value=58 if selected_domain != DomainName.general_outpatient else 28)
        sex = st.selectbox("Sex", options=["male", "female", "other"])
        chief_complaint = st.text_input("Chief complaint", value=preset["chief_complaint"])
        timeline = st.text_area("Timeline", value=preset["timeline"])
    with col2:
        symptoms_text = st.text_area("Symptoms (comma-separated)", value=preset["symptoms"])
        history_text = st.text_area("History (comma-separated)", value=preset["history"])
        notes = st.text_area("Notes", value=preset["notes"])
        spo2 = st.number_input("SpO2", min_value=50, max_value=100, value=preset["spo2"])
        hr = st.number_input("Heart rate", min_value=20, max_value=220, value=preset["hr"])
        sbp = st.number_input("Systolic BP", min_value=40, max_value=250, value=preset["sbp"])
        temperature = st.number_input("Temperature C", min_value=30.0, max_value=43.0, value=float(preset["temperature"]))
    submitted = st.form_submit_button("Run TAO triage")

if submitted:
    orchestrator = build_orchestrator()
    request = CaseRequest(
        case_id=case_id,
        patient=PatientPayload(age=int(age), sex=sex),
        preferred_domain=selected_domain,
        chief_complaint=chief_complaint,
        timeline=timeline,
        symptoms=[item.strip() for item in symptoms_text.split(",") if item.strip()],
        history=[item.strip() for item in history_text.split(",") if item.strip()],
        vitals=VitalsPayload(
            spo2=int(spo2),
            heart_rate=int(hr),
            systolic_bp=int(sbp),
            temperature_c=float(temperature),
        ),
        notes=notes,
    )
    result = orchestrator.run(request)
    st.subheader("Final decision")
    st.json(json.loads(result.model_dump_json()))
