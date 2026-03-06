import streamlit as st
import time
import pandas as pd
from app.core.orchestrator import Orchestrator
from app.models.schemas import PatientCase
from app.models.enums import Tier

# Cấu hình trang
st.set_page_config(page_title="Live Model Comparison", page_icon="🔬", layout="wide")

st.title("🔬 Live Model Comparison (A/B Testing)")
st.markdown("Chạy thử nghiệm một ca bệnh thực tế qua nhiều mô hình LLM khác nhau để so sánh trực tiếp quyết định chẩn đoán, chi phí và thời gian xử lý.")

# 1. Nhập liệu ca bệnh
with st.expander("📝 Nhập liệu Ca bệnh (Patient Case)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        symptoms = st.text_area(
            "Triệu chứng (Symptoms):", 
            "I have excruciating stomach cramps, relentless vomiting, and a bizarre metallic taste in my mouth. My fingers are numb."
        )
    with col2:
        history = st.text_area(
            "Tiền sử (History):", 
            "I started taking a miracle energy powder from a blank silver pouch. I don't know the ingredients."
        )

# 2. Chọn Model để so sánh
st.subheader("⚙️ Cấu hình Models thi đấu")
available_models = [
    "models/gemini-2.5-flash",
    "models/gemini-1.5-pro", 
    "models/gemini-1.5-flash"
]
selected_models = st.multiselect(
    "Chọn các model muốn đưa lên bàn cân:", 
    available_models, 
    default=["models/gemini-2.5-flash", "models/gemini-1.5-pro"]
)

# 3. Nút chạy thử nghiệm
if st.button("🚀 Bắt đầu So sánh (Run Comparison)", type="primary"):
    if not symptoms or len(selected_models) < 2:
        st.warning("⚠️ Vui lòng nhập triệu chứng và chọn ít nhất 2 models để hệ thống có thể so sánh.")
    else:
        # Khởi tạo case bệnh
        case = PatientCase(case_id=str(int(time.time())), symptoms=symptoms, medical_history=history)
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Chạy vòng lặp qua từng model được chọn
        for i, model_name in enumerate(selected_models):
            status_text.text(f"⏳ Đang gọi API thử nghiệm với {model_name}...")
            
            # Khởi tạo Orchestrator mới
            orch = Orchestrator()
            
            # Kỹ thuật Dynamic Override: Ép Orchestrator và LLMClient dùng model đang test
            orch.tier_models = {
                Tier.TIER_1: model_name,
                Tier.TIER_2: model_name,
                Tier.TIER_3: model_name
            }
            orch.llm_client.default_google_model = model_name
            
            start_time = time.time()
            try:
                # Chạy luồng TAO
                state = orch.process_case(case)
                end_time = time.time()
                
                # Trích xuất kết quả
                final_diag = state.final_diagnosis
                
                results.append({
                    "Model": model_name,
                    "Quyết định (Tier cuối)": final_diag.escalation_decision.value if final_diag else "REJECTED",
                    "Tier Dừng lại": state.current_tier.name,
                    "Độ Tự tin": f"{(final_diag.confidence_score * 100):.1f}%" if final_diag else "N/A",
                    "Thời gian (s)": round(end_time - start_time, 2),
                    "Chi phí ($)": round(state.total_cost, 5),
                    "Tokens": state.total_tokens,
                    "Chẩn đoán tóm tắt": final_diag.diagnosis_summary[:120] + "..." if final_diag else "Không có chẩn đoán (Bị từ chối)"
                })
            except Exception as e:
                results.append({
                    "Model": model_name,
                    "Quyết định (Tier cuối)": "LỖI HỆ THỐNG",
                    "Tier Dừng lại": "N/A",
                    "Độ Tự tin": "N/A",
                    "Thời gian (s)": "N/A",
                    "Chi phí ($)": "N/A",
                    "Tokens": 0,
                    "Chẩn đoán tóm tắt": str(e)
                })
            
            # Cập nhật thanh tiến trình
            progress_bar.progress((i + 1) / len(selected_models))
        
        status_text.text("✅ Hoàn tất so sánh thực tế!")
        
        # 4. In bảng kết quả
        st.subheader("📊 Kết quả Thi đấu (Live Metrics)")
        df_results = pd.DataFrame(results)
        
        # Bôi đậm model chạy nhanh nhất
        st.dataframe(
            df_results.style.highlight_min(subset=['Thời gian (s)'], color='#1f77b4', axis=0)
                      .highlight_min(subset=['Chi phí ($)'], color='#2ca02c', axis=0),
            use_container_width=True,
            hide_index=True
        )
        
        st.info("""
        **💡 Cách phân tích kết quả:**
        - **Thời gian (s) & Chi phí ($):** Thông thường các bản `flash` sẽ rẻ và nhanh hơn bản `pro`. (Màu xanh hiển thị model tối ưu nhất).
        - **Quyết định & Tier Dừng lại:** Đây là phần quan trọng nhất. Hãy quan sát xem model nào "dễ dãi" (dừng lại sớm ở Tier 1 và tự tin chẩn đoán) và model nào "khắt khe" (đẩy lên Tier cao hơn hoặc REJECT do thiếu thông tin).
        """)