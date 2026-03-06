import streamlit as st
import pandas as pd

# Cấu hình trang
st.set_page_config(page_title="TAO Benchmarks", page_icon="📊", layout="wide")

st.title("📊 Phân tích & So sánh Hiệu năng (TAO Framework)")
st.markdown("""
Trang này trực quan hóa các số liệu benchmark được trích xuất từ bài báo cáo khoa học: 
**"Tiered Agentic Oversight: A Hierarchical Multi-Agent System for AI Safety in Healthcare"**.
""")

# --- PHẦN 1: SO SÁNH KIẾN TRÚC ---
st.header("1. So sánh Kiến trúc Hệ thống An toàn AI")
st.markdown("So sánh TAO với các phương pháp tổ chức Agent và giám sát truyền thống (Dữ liệu từ Table 1).")

arch_data = {
    "Tiêu chí": [
        "Sự đa dạng Agent (Agent Diversity)", 
        "Cơ chế phát hiện lỗi (Error Detection)", 
        "Chiến lược sửa lỗi (Mitigation Strategy)", 
        "Rủi ro thất bại (Failure Risk)", 
        "Khả năng thích ứng (Adaptability)", 
        "Khả năng mở rộng (Scalability)", 
        "Tính minh bạch (Transparency)", 
        "Mô hình giao tiếp (Conv. Pattern)"
    ],
    "TAO (Đề xuất)": ["Có", "Đánh giá theo tầng", "Báo cáo vượt cấp", "Thấp", "Cao", "Trung bình", "Cao", "Linh hoạt"],
    "MedAgents": ["Có", "Agent chéo", "Tự tinh chỉnh", "Trung bình", "Trung bình", "Trung bình", "Trung bình", "Cố định"],
    "Voting": ["Có", "Bỏ phiếu", "Theo số đông", "Trung bình", "Thấp", "Trung bình", "Trung bình", "Cố định"],
    "Single LLM": ["Không", "Đánh giá 1 lần", "Không có", "Cao", "Không có", "Cao", "Thấp", "Cố định"],
    "Human Oversight": ["Có", "Con người đánh giá", "Con người sửa", "Rất thấp", "Cao", "Thấp", "TB - Cao", "Tương tác"]
}

df_arch = pd.DataFrame(arch_data)
st.dataframe(df_arch, use_container_width=True, hide_index=True)

st.info("💡 **Insights:** TAO khắc phục được rủi ro thất bại cao của Single LLM, đồng thời duy trì khả năng mở rộng tốt hơn nhiều so với việc phụ thuộc hoàn toàn vào con người (Human Oversight).")

st.divider()

# --- PHẦN 2: HIỆU NĂNG LLM ---
st.header("2. Hiệu năng các Mô hình Ngôn ngữ (LLMs)")
st.markdown("Đánh giá kiến trúc TAO khi chạy trên các Model khác nhau qua 5 bộ dữ liệu Y khoa (Table 2, 5, 6).")

llm_data = {
    "Bộ dữ liệu (Benchmark)": [
        "MedSafetyBench (Đạo đức)", 
        "Red Teaming (Bẻ khóa)", 
        "SafetyBench (An toàn chung)", 
        "Medical Triage (Phân loại)", 
        "MM-Safety (Đa phương thức)"
    ],
    "Gemini 2.5 Pro": [4.85, 64.6, 92.0, 62.0, 90.3],
    "Gemini 2.0 Flash": [4.88, 58.3, 93.4, 57.9, 80.0],
    "OpenAI o3": [4.89, 55.1, 90.1, 62.2, 70.1]
}

df_llm = pd.DataFrame(llm_data)

col1, col2 = st.columns([1, 1.5])

with col1:
    st.dataframe(df_llm, use_container_width=True, hide_index=True)
    st.caption("Điểm số càng cao càng an toàn. Dữ liệu có dung sai (±) đã được lược bỏ để dễ nhìn.")

with col2:
    # Trực quan hóa bằng biểu đồ cột
    df_chart = df_llm.set_index("Bộ dữ liệu (Benchmark)")
    st.bar_chart(df_chart)

st.success("🏆 **Kết luận Kỹ thuật:** Các model nhẹ và nhanh (như Gemini Flash) khi kết hợp với quy trình phân tầng chặt chẽ của TAO có thể đạt độ an toàn tương đương (hoặc thậm chí nhỉnh hơn) các model nặng ở các tác vụ văn bản thuần túy.")