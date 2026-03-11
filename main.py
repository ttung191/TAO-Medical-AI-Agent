import streamlit as st
import time
import os
import google.generativeai as genai
from app.core.orchestrator import Orchestrator
from app.models.schemas import PatientCase

# Cấu hình trang (Phải đặt ở dòng đầu tiên)
st.set_page_config(page_title="TAO Medical AI", page_icon="🏥", layout="wide")

# --- QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
# Lưu trữ dữ liệu ca bệnh và lịch sử chat để không bị mất khi màn hình load lại
if "medical_context" not in st.session_state:
    st.session_state.medical_context = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "case_processed" not in st.session_state:
    st.session_state.case_processed = False

# Giao diện chính
st.title("🏥 TAO: Tiered Agentic Oversight")
st.markdown("Hệ thống Trí tuệ Nhân tạo Y tế Đa Tác nhân (Multi-Agent) hỗ trợ chẩn đoán lâm sàng an toàn.")

# --- PHẦN 1: NHẬP LIỆU ---
with st.expander("📝 THÔNG TIN LÂM SÀNG (CLINICAL DATA)", expanded=not st.session_state.case_processed):
    col1, col2 = st.columns(2)
    with col1:
        symptoms = st.text_area("Triệu chứng (Symptoms):", height=150, 
                                placeholder="VD: Bệnh nhân nam 55 tuổi, đau thắt ngực trái...")
    with col2:
        history = st.text_area("Tiền sử bệnh (Medical History):", height=150, 
                               placeholder="VD: Cao huyết áp, tiểu đường tuýp 2, dị ứng Penicillin...")
        
    run_btn = st.button("🚀 Bắt đầu Hội chẩn (Run TAO)", type="primary", use_container_width=True)

# --- PHẦN 2: XỬ LÝ LƯỢNG TAO ---
if run_btn:
    if not symptoms:
        st.warning("⚠️ Vui lòng nhập Triệu chứng của bệnh nhân!")
    else:
        # Reset lại chat khi chạy ca mới
        st.session_state.chat_history = []
        st.session_state.case_processed = True
        
        with st.spinner("⏳ Hệ thống đang phân tích phân tầng (Triage -> Specialist -> Consultant)..."):
            try:
                # Khởi tạo ca bệnh và Orchestrator
                case = PatientCase(case_id=str(int(time.time())), symptoms=symptoms, medical_history=history)
                orch = Orchestrator()
                
                start_time = time.time()
                state = orch.process_case(case) # Chạy luồng
                latency = time.time() - start_time
                
                # Lấy kết quả cuối cùng
                final_diag = state.final_diagnosis
                
                # Lưu Context lại để tí nữa Chatbox có thể đọc được
                st.session_state.medical_context = f"""
                -- THÔNG TIN CA BỆNH --
                Triệu chứng: {symptoms}
                Tiền sử: {history}
                
                -- KẾT QUẢ TAO CHẨN ĐOÁN --
                Quyết định: {final_diag.escalation_decision.value if final_diag else 'REJECTED'}
                Chẩn đoán: {final_diag.diagnosis_summary if final_diag else 'Không có (Thiếu thông tin)'}
                Kế hoạch: {final_diag.treatment_plan if final_diag else 'Không có'}
                """
                
                # --- HIỂN THỊ DASHBOARD KẾT QUẢ CHÍNH ---
                st.success("✅ Đã hoàn tất hội chẩn!")
                
                # 1. Trực quan hóa các Metrics (Chỉ số)
                st.markdown("### 📊 Thông số Hệ thống")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tier Dừng lại", state.current_tier.name)
                m2.metric("Độ Tự tin (Confidence)", f"{(final_diag.confidence_score * 100):.1f}%" if final_diag else "N/A")
                m3.metric("Thời gian xử lý", f"{latency:.2f} s")
                m4.metric("Chi phí API", f"${state.total_cost:.5f}")
                
                # 2. Hiển thị Kết luận y khoa
                st.markdown("### 🏁 Chẩn đoán Cuối cùng (Final Medical Opinion)")
                if final_diag:
                    if final_diag.escalation_decision.value == "REJECT":
                        st.error(f"⚠️ HỆ THỐNG TỪ CHỐI CHẨN ĐOÁN:\n\n{final_diag.diagnosis_summary}")
                    else:
                        st.info(f"**Tóm tắt:** {final_diag.diagnosis_summary}")
                        st.write(f"**Hướng xử trí:** {final_diag.treatment_plan}")
                
                with st.expander("📑 Xem chi tiết Lịch sử suy luận (Reasoning Logs)"):
                    st.json(state.model_dump())

            except Exception as e:
                st.error(f"Lỗi hệ thống: {str(e)}")

st.divider()

# --- PHẦN 3: CHATBOX (INTERACTIVE FOLLOW-UP) ---
if st.session_state.case_processed and st.session_state.medical_context:
    st.subheader("💬 Trợ lý Y khoa (Hỏi đáp chuyên sâu)")
    st.caption("Hãy đặt câu hỏi về ca bệnh này (VD: *Giải thích rõ hơn về tương tác thuốc?* hoặc *Nếu bệnh nhân bị dị ứng thì đổi thuốc gì?*)")
    
    # Render lịch sử chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập liệu chat
    if prompt := st.chat_input("Hỏi thêm hệ thống về ca bệnh này..."):
        # 1. Hiển thị tin nhắn của user
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Gửi cho LLM (Gemini) để lấy câu trả lời
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Cấu hình Gemini API cho phần Chat
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('gemini-2.5-flash') # Dùng bản flash cho chat nhanh
                
                # Xây dựng Prompt bối cảnh: Ép AI đóng vai bác sĩ giải thích dựa trên kết quả TAO
                system_instruction = f"""
                Bạn là một Bác sĩ Trưởng khoa lão luyện. Dưới đây là hồ sơ bệnh án và kết luận chẩn đoán của hệ thống (TAO).
                Dựa vào bối cảnh này, hãy trả lời câu hỏi của người dùng một cách ngắn gọn, chuyên nghiệp và chính xác.
                
                {st.session_state.medical_context}
                """
                
                # Ghép lịch sử chat cũ vào để AI nhớ ngữ cảnh
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[:-1]])
                full_prompt = f"{system_instruction}\n\nLịch sử trò chuyện:\n{history_text}\n\nUser: {prompt}\nBác sĩ:"
                
                # Gọi API
                response = model.generate_content(full_prompt)
                full_response = response.text
                
                # Hiển thị
                message_placeholder.markdown(full_response)
                
                # Lưu vào lịch sử
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                message_placeholder.error(f"Lỗi kết nối AI: {str(e)}")
