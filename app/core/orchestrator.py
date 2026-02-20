import time
from app.models.schemas import PatientCase, SystemState, AgentDiagnosis, CostMetrics
from app.models.enums import Tier, AgentStatus, EscalationDecision, RiskLevel
from app.core.safety import SafetyFilter
from app.core.recruiter import AgentRecruiter
from app.agents.medical_agent import DynamicMedicalAgent
from app.core.llm_client import LLMClient

class Orchestrator:
    def __init__(self):
        self.safety = SafetyFilter()
        self.recruiter = AgentRecruiter()
        self.llm_client = LLMClient()
        
        # Dùng cứng Model Gemini 2.5 Flash (Mới nhất & Ổn định)
        base_model = "models/gemini-2.5-flash"
        
        self.tier_models = {
            Tier.TIER_1: base_model,
            Tier.TIER_2: base_model,
            Tier.TIER_3: base_model
        }

    def process_case(self, case: PatientCase) -> SystemState:
        state = SystemState(case=case, current_tier=Tier.TIER_1, status=AgentStatus.WORKING)
        start_time = time.time()

        # 1. Safety Check
        if self.safety.assess_input_risk(case.symptoms) == RiskLevel.CRITICAL:
            state.status = AgentStatus.FAILED
            state.logs.append("🚨 Safety Violation Detected.")
            return state

        # 2. Dynamic Recruitment
        state.logs.append("🔍 Recruiting Medical Team...")
        try:
            recruitment_plan = self.recruiter.recruit_and_route(f"{case.symptoms} {case.medical_history}")
            state.recruited_agents = recruitment_plan
            state.logs.append(f"📋 Team Assembled: {recruitment_plan}")
        except Exception as e:
            state.logs.append(f"⚠️ Recruitment failed: {e}. Using Default Team.")
            state.recruited_agents = {
                Tier.TIER_1: ["General_Nurse"],
                Tier.TIER_2: ["General_Practitioner"],
                Tier.TIER_3: ["Senior_Consultant"]
            }

        # 3. Main Loop
        max_hops = 6 
        hops = 0
        current_feedback = None 
        tier_order = [Tier.TIER_1, Tier.TIER_2, Tier.TIER_3]
        
        while hops < max_hops and state.status == AgentStatus.WORKING:
            hops += 1
            current_tier_idx = tier_order.index(state.current_tier)
            
            role_names = state.recruited_agents.get(state.current_tier)
            if not role_names:
                role_names = ["Generalist_Doctor"]
            
            state.logs.append(f"--- Running {state.current_tier.value} (Attempt {hops}) ---")
            
            # Tạo agents
            tier_agents = [
                DynamicMedicalAgent(
                    model_name=self.tier_models[state.current_tier],
                    tier=state.current_tier,
                    role_name=role
                ) for role in role_names
            ]
            
            prev_context = self._get_previous_context(state, current_tier_idx)
            
            # Chạy agents
            tier_outputs = []
            for agent in tier_agents:
                try:
                    diag = agent.process_case(case, prev_context, feedback_received=current_feedback)
                    tier_outputs.append(diag)
                    self._update_cost(state, diag)
                except Exception as e:
                    state.logs.append(f"❌ Agent {agent.role_name} failed: {e}")

            # Xử lý lỗi danh sách rỗng (Tránh lỗi max() arg is empty)
            if not tier_outputs:
                state.logs.append("⚠️ No agents returned a valid diagnosis. Retrying or Aborting.")
                if hops >= max_hops:
                    state.status = AgentStatus.FAILED
                continue
            
            # Chọn kết quả tốt nhất
            best_diagnosis = max(tier_outputs, key=lambda x: x.confidence_score)
            
            state.interaction_history.append(best_diagnosis)
            state.logs.append(f"✅ {best_diagnosis.agent_name}: {best_diagnosis.escalation_decision}")

            # Decision Logic
            decision = best_diagnosis.escalation_decision
            
            if decision == EscalationDecision.REJECT and current_tier_idx > 0:
                state.current_tier = tier_order[current_tier_idx - 1]
                current_feedback = best_diagnosis.feedback_to_lower_tier
                state.logs.append(f"🔙 REJECTED! Feedback: {current_feedback}")
                state.status = AgentStatus.WORKING
                
            elif decision == EscalationDecision.ESCALATE and current_tier_idx < 2:
                state.current_tier = tier_order[current_tier_idx + 1]
                current_feedback = None
                state.logs.append(f"🔼 Escalating to {state.current_tier}")
                
            else:
                state.final_diagnosis = best_diagnosis
                state.status = AgentStatus.COMPLETED
                
        state.total_time_ms = (time.time() - start_time) * 1000
        return state

    def _get_previous_context(self, state, current_idx):
        if current_idx == 0: return None
        context = ""
        for diag in state.interaction_history[-3:]:
            context += f"[{diag.agent_name}]: {diag.diagnosis_summary}\n"
        return context

    def _update_cost(self, state, diag):
        state.total_cost += diag.metrics.total_cost_usd
        state.total_tokens += (diag.metrics.input_tokens + diag.metrics.output_tokens)