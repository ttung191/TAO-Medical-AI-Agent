import streamlit as st
import requests
import json

st.set_page_config(page_title="TAO Medical AI", layout="wide")
st.title("TAO Medical AI - Clinical Dashboard")

API_URL = "http://localhost:8000/v2/cases/analyze"

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader(" Bệnh án (Intake)")
    if st.button("Ca Mẫu: Đau ngực"):
        st.session_state.update({"complaint": "Đau thắt ngực lan ra tay", "sym": "Khó thở, vã mồ hôi", "hist": "THA"})
        
    complaint = st.text_input("Lý do khám", value=st.session_state.get("complaint", ""))
    symptoms = st.text_area("Triệu chứng", value=st.session_state.get("sym", ""))
    history = st.text_area("Tiền sử", value=st.session_state.get("hist", ""))
    
    if st.button(" Chạy Hội Chẩn Đa Tầng", type="primary"):
        if len(complaint) > 4:
            payload = {
                "chief_complaint": complaint,
                "symptoms": [s.strip() for s in symptoms.split(",")] if symptoms else [],
                "medical_history": [h.strip() for h in history.split(",")] if history else [],
                "current_medications": [], "vitals": {}, "labs": {}
            }
            with st.spinner("Các Agent đang hội chẩn..."):
                res = requests.post(API_URL, json=payload)
                if res.status_code == 200:
                    st.session_state["result"] = res.json()
                else:
                    st.error(f"Lỗi API: {res.text}")

with col2:
    st.subheader("Kết quả (TAO Output)")
    if "result" in st.session_state:
        data = st.session_state["result"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tuyến kết thúc", f"Tier {data['final_tier']}")
        c2.metric("Mức rủi ro", data['risk'].upper())
        c3.metric("Hướng xử trí", data['disposition'].replace("_", " ").title())
        
        st.info(f"**Chẩn đoán:** {data['diagnosis']}")
        
        st.markdown("#### Đội ngũ AI")
        st.write(", ".join([f"Tier {t['tier']} ({t['role']})" for t in data['recruited_team']]))
        
        st.markdown("#### Lưu vết (Audit Trail)")
        for step in data['audit_trail']:
            with st.expander(f"Tier {step['tier']} - {step['role']}", expanded=True):
                st.write(step['rationale'])
                
        st.caption(f"Token đã dùng: {data['performance_metrics']['total_tokens']} (~ ${data['performance_metrics']['cost_usd']})")