import streamlit as st
import pandas as pd
import plotly.express as px
from app.models.schemas import SystemState, AgentDiagnosis
from app.models.enums import Tier, RiskLevel

def render_sidebar():
    """Hiển thị bảng điều khiển bên trái"""
    with st.sidebar:
        st.title("🏥 TAO System Monitor")
        
        # 1. Total Stats (Lấy từ session state)
        if "total_cost" not in st.session_state:
            st.session_state.total_cost = 0.0
        if "total_tokens" not in st.session_state:
            st.session_state.total_tokens = 0
            
        col1, col2 = st.columns(2)
        col1.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
        col2.metric("Tokens", f"{st.session_state.total_tokens}")
        
        st.divider()
        
        # 2. Settings
        st.subheader("⚙️ Configuration")
        st.toggle("Debug Mode", value=False, key="debug_mode")
        st.info("Medical AI System based on Tiered Agentic Oversight (TAO) Framework.")

def render_diagnosis_card(diagnosis: AgentDiagnosis, expanded: bool = False):
    """Hiển thị thẻ kết quả chẩn đoán của một Agent"""
    
    # Màu sắc dựa trên Risk Level
    color_map = {
        RiskLevel.LOW: "green",
        RiskLevel.MODERATE: "orange",
        RiskLevel.HIGH: "red",
        RiskLevel.CRITICAL: "red"
    }
    color = color_map.get(diagnosis.risk_assessment, "grey")
    
    with st.expander(f"{diagnosis.agent_name} Report (Risk: {diagnosis.risk_assessment})", expanded=expanded):
        st.markdown(f"**🩺 Diagnosis:** {diagnosis.diagnosis_summary}")
        st.markdown(f"**💊 Treatment:** {diagnosis.treatment_plan}")
        
        # Hiển thị độ tin cậy dưới dạng progress bar
        st.progress(diagnosis.confidence_score, text=f"Confidence Score: {diagnosis.confidence_score*100:.1f}%")
        
        st.markdown(f"**🤔 Reasoning:**")
        st.info(diagnosis.reasoning)
        
        # Metrics nhỏ
        st.caption(f"⚡ Latency: {diagnosis.metrics.latency_ms:.0f}ms | 💰 Cost: ${diagnosis.metrics.total_cost_usd:.5f}")

def render_final_result(state: SystemState):
    """Hiển thị kết luận cuối cùng đẹp mắt"""
    st.divider()
    st.subheader("🏁 Final Medical Opinion")
    
    if state.final_diagnosis:
        diag = state.final_diagnosis
        
        # Alert box lớn dựa trên mức độ nguy hiểm
        if diag.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            st.error(f"⚠️ **CRITICAL WARNING**: {diag.diagnosis_summary}")
        else:
            st.success(f"✅ **CONCLUSION**: {diag.diagnosis_summary}")
            
        st.markdown(f"**Recommended Action:** {diag.treatment_plan}")
        st.markdown(f"*Diagnosed by: {diag.agent_name}*")
    else:
        st.error("System could not reach a conclusion.")