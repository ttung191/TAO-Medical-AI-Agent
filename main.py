import streamlit as st
import time
from app.core.orchestrator import Orchestrator
from app.models.schemas import PatientCase
from app.models.enums import Tier, EscalationDecision
from app.ui.layout import (
    setup_page, 
    render_sidebar, 
    render_header, 
    render_diagnosis_card, 
    render_final_result,
    render_download_button
)

# 1. Setup
setup_page()
render_sidebar()
render_header()

# 2. Init Logic
@st.cache_resource
def get_orchestrator():
    return Orchestrator()

orchestrator = get_orchestrator()

# 3. Session State Init
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# 4. Input Form
with st.form("patient_form"):
    col1, col2 = st.columns([2, 1])
    with col1:
        symptoms = st.text_area("Triệu chứng (Symptoms):", height=100, 
                               placeholder="Ví dụ: Đau đầu dữ dội, cứng cổ, sốt cao...")
    with col2:
        history = st.text_area("Tiền sử (History):", height=100, 
                              placeholder="Ví dụ: Vừa đi du lịch về...")
    
    submitted = st.form_submit_button("🚀 Bắt đầu Chẩn đoán (TAO 2.0)", use_container_width=True)

# 5. Logic xử lý
if submitted:
    if not symptoms:
        st.warning("⚠️ Vui lòng nhập triệu chứng!")
    else:
        # Reset kết quả cũ
        st.session_state.current_result = None
        
        case = PatientCase(case_id=str(int(time.time())), symptoms=symptoms, medical_history=history)
        
        # Chạy tiến trình
        status_box = st.status("🔍 Hệ thống đang phân tích...", expanded=True)
        try:
            # Container để hiển thị log realtime
            log_container = status_box.container()
            
            # Chạy Orchestrator
            result_state = orchestrator.process_case(case)
            
            # Hiển thị log vắn tắt
            for log in result_state.logs:
                log_container.write(f"👉 {log}")
            
            st.session_state.current_result = result_state
            
            # Update metrics global
            if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0
            if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
            
            st.session_state.total_cost += result_state.total_cost
            st.session_state.total_tokens += result_state.total_tokens
            
            status_box.update(label="✅ Phân tích hoàn tất!", state="complete", expanded=False)
            
        except Exception as e:
            status_box.update(label="❌ Lỗi hệ thống!", state="error")
            st.error(f"Error Details: {e}")

# 6. Hiển thị kết quả (Logic mới dùng interaction_history)
if st.session_state.current_result:
    result = st.session_state.current_result
    
    # A. Kết luận cuối cùng
    render_final_result(result)
    
    # B. Chi tiết quá trình (History)
    st.subheader("📑 Interaction History & Reasoning")
    st.caption("Dưới đây là toàn bộ quá trình hội chẩn, bao gồm cả các vòng lặp phản hồi (nếu có).")
    
    # Lặp qua lịch sử (Đảo ngược để cái mới nhất lên đầu)
    if result.interaction_history:
        for i, diag in enumerate(reversed(result.interaction_history)):
            # Xác định trạng thái thẻ (mở rộng cái đầu tiên)
            is_expanded = (i == 0)
            
            # Hiển thị thêm thông tin Decision (Escalate/Reject)
            decision_icon = "✅"
            if diag.escalation_decision == EscalationDecision.ESCALATE: decision_icon = "🔼"
            elif diag.escalation_decision == EscalationDecision.REJECT: decision_icon = "🔙"
            
            # Render thẻ chuẩn
            render_diagnosis_card(diag, expanded=is_expanded)
            
            # Hiển thị Feedback nếu có (khi bị Reject)
            if diag.escalation_decision == EscalationDecision.REJECT and diag.feedback_to_lower_tier:
                st.warning(f"🔙 **REJECTION FEEDBACK:** {diag.feedback_to_lower_tier}")
                
            # Kẻ dòng ngăn cách các lượt
            if i < len(result.interaction_history) - 1:
                st.markdown("---")
    else:
        st.info("Chưa có dữ liệu chẩn đoán chi tiết.")

    # C. Debug Logs (Ẩn trong Expander)
    with st.expander("🛠️ System Debug Logs"):
        for log in result.logs:
            st.text(log)

    # D. Nút Download
    render_download_button(result)