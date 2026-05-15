import streamlit as st
import requests
import json

# Cấu hình trang
st.set_page_config(page_title="TAO Medical AI", layout="wide")
st.title("TAO Medical AI - Clinical Dashboard")

API_URL = "http://localhost:8000/v2/cases/analyze"

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📝 Bệnh án (Intake)")
    
    # Nút bấm nạp nhanh Case Study 1 (Hóc búa)
    if st.button("🚨 Nạp Ca Test: Đau ngực ẩn / Sốc phản vệ"):
        st.session_state.update({
            "age": 72, "sex": "Nam",
            "complaint": "Tôi bị ngứa ngáy khắp người, nổi mẩn đỏ sau khi ăn hải sản hôm qua. Sáng nay tỉnh dậy thấy hơi đau ê ẩm ở ngực dưới và sau lưng.",
            "sym": "Ngứa, nổi mề đay, mất ngủ, đau ngực dưới, vã mồ hôi lạnh nhẹ", 
            "hist": "Đái tháo đường type 2 (15 năm), Rối loạn lipid máu",
            "meds": "Metformin",
            "hr": "115", "bp": "85/50", "spo2": "96", "labs": ""
        })
        
    # --- FORM NHẬP LIỆU ---
    r1_c1, r1_c2 = st.columns(2)
    age = r1_c1.number_input("Tuổi", min_value=0, max_value=120, value=st.session_state.get("age", 45))
    sex = r1_c2.selectbox("Giới tính", ["Nam", "Nữ", "Khác"], index=["Nam", "Nữ", "Khác"].index(st.session_state.get("sex", "Nam")))
    
    complaint = st.text_area("Lý do khám (Chief Complaint)*", value=st.session_state.get("complaint", ""), height=100)
    symptoms = st.text_input("Triệu chứng", value=st.session_state.get("sym", ""))
    history = st.text_input("Tiền sử", value=st.session_state.get("hist", ""))
    medications = st.text_input("Thuốc đang dùng", value=st.session_state.get("meds", ""))
    
    # Expander cho Sinh hiệu & Xét nghiệm để UI gọn gàng
    with st.expander("🩺 Sinh hiệu & Xét nghiệm (Vitals & Labs)", expanded=True):
        v1, v2, v3 = st.columns(3)
        hr = v1.text_input("Nhịp tim (bpm)", value=st.session_state.get("hr", ""))
        bp = v2.text_input("Huyết áp (mmHg)", value=st.session_state.get("bp", ""))
        spo2 = v3.text_input("SpO2 (%)", value=st.session_state.get("spo2", ""))
        labs = st.text_area("Cận lâm sàng", value=st.session_state.get("labs", ""))
    
    # Nút submit
    if st.button("🚀 Chạy Hội Chẩn Đa Tầng", type="primary", use_container_width=True):
        if len(complaint) > 4:
            # Gom Vitals thành Dict
            vitals_dict = {}
            if hr: vitals_dict["HR"] = hr
            if bp: vitals_dict["BP"] = bp
            if spo2: vitals_dict["SpO2"] = spo2
            
            # Đóng gói Payload chuẩn theo Backend
            payload = {
                "patient_name": "Ẩn danh", # PHI filter sẽ lo phần này
                "age": age,
                "patient_sex": sex,
                "chief_complaint": complaint,
                "symptoms": [s.strip() for s in symptoms.split(",")] if symptoms else [],
                "medical_history": [h.strip() for h in history.split(",")] if history else [],
                "current_medications": [m.strip() for m in medications.split(",")] if medications else [],
                "vitals": vitals_dict,
                "labs": {"notes": labs} if labs else {}
            }
            
            with st.spinner("Đội ngũ AI đang phân tích dữ liệu..."):
                try:
                    res = requests.post(API_URL, json=payload)
                    if res.status_code == 200:
                        st.session_state["result"] = res.json()
                    else:
                        st.error(f"Lỗi API ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Không thể kết nối đến Backend: {e}")
        else:
            st.warning("Vui lòng nhập Lý do khám chi tiết hơn (ít nhất 5 ký tự).")

with col2:
    st.subheader("📊 Kết quả (TAO Output)")
    if "result" in st.session_state:
        data = st.session_state["result"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tuyến kết thúc", f"Tier {data.get('final_tier', 'N/A')}")
        c2.metric("Mức rủi ro", str(data.get('risk', 'N/A')).upper())
        c3.metric("Hướng xử trí", str(data.get('disposition', 'N/A')).replace("_", " ").title())
        
        st.info(f"**Chẩn đoán:** {data.get('diagnosis', 'N/A')}")
        
        st.markdown("#### Đội ngũ AI")
        if 'recruited_team' in data:
            st.write(", ".join([f"Tier {t['tier']} ({t['role']})" for t in data['recruited_team']]))
        
        st.markdown("#### Lưu vết (Audit Trail)")
        if 'audit_trail' in data:
            for step in data['audit_trail']:
                with st.expander(f"Tier {step.get('tier')} - {step.get('role')}", expanded=True):
                    st.write(step.get('rationale', ''))
                
        if 'performance_metrics' in data:
            perf = data['performance_metrics']
            st.caption(f"Token đã dùng: {perf.get('total_tokens', 0)} (~ ${perf.get('cost_usd', 0)})")