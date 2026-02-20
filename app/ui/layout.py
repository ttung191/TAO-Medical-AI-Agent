import streamlit as st
import time
from app.models.schemas import SystemState, AgentDiagnosis
from app.models.enums import Tier, RiskLevel
from app.utils.report_generator import create_pdf_report

def setup_page():
    """Cấu hình trang cơ bản và CSS"""
    st.set_page_config(
        page_title="Medical TAO System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS để giao diện đẹp hơn
    st.markdown("""
        <style>
        .streamlit-expanderHeader { background-color: #f8f9fa; border-radius: 5px; font-weight: bold; }
        div[data-testid="stStatusWidget"] { border: 1px solid #ddd; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .stAlert { border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Vẽ thanh bên trái (Sidebar)"""
    with st.sidebar:
        st.title("🏥 TAO Monitor")
        st.caption("System Metrics Dashboard")
        
        # 1. Metrics (Lấy từ Session State)
        if "total_cost" not in st.session_state:
            st.session_state.total_cost = 0.0
        if "total_tokens" not in st.session_state:
            st.session_state.total_tokens = 0
            
        # Hiển thị dạng bảng số liệu
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Cost", value=f"${st.session_state.total_cost:.4f}")
        with col2:
            st.metric(label="Tokens", value=f"{st.session_state.total_tokens}")
        
        st.divider()
        
        # 2. Settings & Info
        st.subheader("⚙️ Configuration")
        debug_mode = st.toggle("Debug Mode", value=False, help="Hiển thị log chi tiết của từng Agent")
        
        st.info(
            """
            **Architecture:**
            - **Tier 1:** Nurse (Triage)
            - **Tier 2:** GP (Diagnosis)
            - **Tier 3:** Specialist (Critical)
            """
        )
        return debug_mode

def render_header():
    """Vẽ tiêu đề chính"""
    st.title("🩺 Medical AI Assistant (TAO)")
    st.markdown("### Tiered Agentic Oversight Framework")
    st.markdown("---")

def render_diagnosis_card(diagnosis: AgentDiagnosis, expanded: bool = False):
    """Vẽ thẻ báo cáo chi tiết của từng Agent"""
    if not diagnosis:
        return

    # Chọn icon dựa trên Tier
    icon_map = {
        Tier.TIER_1: "👩‍⚕️ Nurse",
        Tier.TIER_2: "👨‍⚕️ GP",
        Tier.TIER_3: "🎓 Specialist"
    }
    
    # Chọn màu dựa trên Risk Level (dùng icon/emoji để biểu thị màu sắc trong text)
    risk_icon = {
        RiskLevel.LOW: "🟢",
        RiskLevel.MODERATE: "jq", # Vàng
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRITICAL: "🔴"
    }
    
    label = f"{icon_map.get(diagnosis.tier, '🤖 Agent')} Report | Risk: {diagnosis.risk_assessment} {risk_icon.get(diagnosis.risk_assessment, '')}"

    with st.expander(label, expanded=expanded):
        st.markdown(f"**🩺 Diagnosis:** {diagnosis.diagnosis_summary}")
        st.markdown(f"**💊 Treatment:** {diagnosis.treatment_plan}")
        
        # Thanh độ tự tin
        st.progress(diagnosis.confidence_score, text=f"Confidence Score: {diagnosis.confidence_score*100:.1f}%")
        
        st.markdown("**🤔 Reasoning:**")
        st.info(diagnosis.reasoning)
        
        # Metrics chân trang
        st.caption(f"⚡ Latency: {diagnosis.metrics.latency_ms:.0f}ms | 💰 Cost: ${diagnosis.metrics.total_cost_usd:.5f} | 🧠 Model: {diagnosis.metrics.model_name}")

def render_final_result(state: SystemState):
    """Vẽ kết luận cuối cùng (Hộp màu to)"""
    st.divider()
    st.subheader("🏁 Final Medical Opinion")
    
    if state.final_diagnosis:
        diag = state.final_diagnosis
        
        # Alert box lớn dựa trên mức độ nguy hiểm
        if diag.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            container = st.error
            icon = "⚠️"
        elif diag.risk_assessment == RiskLevel.MODERATE:
            container = st.warning
            icon = "⚖️"
        else:
            container = st.success
            icon = "✅"
            
        container(
            f"**{icon} CONCLUSION ({diag.risk_assessment}):** {diag.diagnosis_summary}\n\n"
            f"**Recommended Action:** {diag.treatment_plan}"
        )
        st.caption(f"Final diagnosis provided by: **{diag.agent_name}**")
    else:
        st.error("❌ System could not reach a conclusion. Please check input data.")

def render_download_button(state: SystemState):
    """Vẽ nút tải PDF"""
    if state.final_diagnosis:
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col2:
            # Gọi hàm tạo PDF từ utils
            try:
                pdf_bytes = create_pdf_report(state.case, state)
                st.download_button(
                    label="📄 Download Medical Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"medical_report_{state.case.case_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")