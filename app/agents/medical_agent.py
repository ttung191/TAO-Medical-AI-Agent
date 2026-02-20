import yaml
import os
from abc import ABC, abstractmethod
from app.core.llm_client import LLMClient
from app.core.knowledge import knowledge_engine
from app.models.schemas import PatientCase, AgentDiagnosis
from app.models.enums import Tier, EscalationDecision, RiskLevel

# --- PHẦN 1: BASE AGENT ---
class BaseAgent(ABC):
    def __init__(self, model_name: str, tier: Tier):
        self.llm = LLMClient()
        self.model_name = model_name
        self.tier = tier
        self.prompts_config = self._load_prompts()

    def _load_prompts(self):
        path = "config/prompts.yaml"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    @abstractmethod
    def get_agent_key(self) -> str:
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    def _safe_get_string(self, data: dict, keys: list, default: str = "") -> str:
        val = None
        for k in keys:
            if k in data:
                val = data[k]
                break
        if val is None: return default
        if isinstance(val, list): return "\n".join([str(v) for v in val])
        return str(val)

# --- PHẦN 2: DYNAMIC AGENT (STRICT MODE) ---
class DynamicMedicalAgent(BaseAgent):
    def __init__(self, model_name: str, tier: Tier, role_name: str):
        super().__init__(model_name, tier)
        self.role_name = role_name 

    def get_agent_key(self) -> str:
        return "dynamic_agent"

    def get_system_prompt(self) -> str:
        # Prompt được nâng cấp để "Khó tính" hơn
        return f"""
        ROLE: You are a {self.role_name} operating at {self.tier.value}.
        
        TIER RESPONSIBILITIES:
        - TIER 1: Screening, triage. Gather ALL vital info.
        - TIER 2: Specialist diagnosis. YOU MUST BE SKEPTICAL.
        - TIER 3: Final Consultant. Validate everything.
        
        !!! STRICT VALIDATION RULES (READ CAREFULLY) !!!
        1. IF specific details are vague (e.g., "travel abroad" without country, "took meds" without name), you MUST DECIDE: "REJECT".
        2. DO NOT GUESS. If information is critical for a differential diagnosis (like Travel History for fever), demand it back.
        3. Better to REJECT now than to misdiagnose later.
        
        YOUR GOAL: Diagnose accurately ONLY IF sufficient data exists. Otherwise, send it back.
        
        OUTPUT FORMAT (JSON):
        - diagnosis_summary (String)
        - treatment_plan (String)
        - reasoning (String)
        - risk_assessment (LOW/MODERATE/HIGH/CRITICAL)
        - confidence_score (Float 0.0-1.0)
        - decision: "RESOLVE", "ESCALATE", or "REJECT"
        - feedback: (String - Required if REJECT) "Ask patient specifically about..."
        """

    def process_case(self, case: PatientCase, previous_reports: str = None, feedback_received: str = None) -> AgentDiagnosis:
        # 1. RAG Lite Check
        combined_text = f"{case.symptoms} {case.medical_history or ''}"
        drug_warnings = knowledge_engine.check_drug_interactions(combined_text)
        
        external_context = ""
        if drug_warnings:
            external_context += "\n[ALERT] DATABASE WARNINGS:\n" + "\n".join(drug_warnings)

        # 2. Build Prompt
        user_prompt = f"CASE: {case.symptoms}\nHISTORY: {case.medical_history}\n"
        
        if external_context:
            user_prompt += f"{external_context}\n"
        
        if previous_reports:
            user_prompt += f"\n--- REPORTS FROM LOWER TIERS ---\n{previous_reports}\n"
            
        if feedback_received:
            user_prompt += f"\n--- FEEDBACK/REJECTION FROM UPPER TIER ---\n⚠️ FIX THIS: {feedback_received}\n"

        user_prompt += "\nOutput valid JSON based on the rules above."

        # 3. Call LLM
        response, metrics = self.llm.generate_json(
            model=self.model_name,
            system_prompt=self.get_system_prompt(),
            user_prompt=user_prompt
        )

        # 4. Return
        return AgentDiagnosis(
            tier=self.tier,
            agent_name=f"{self.role_name}_({self.tier.name})",
            role=self.role_name,
            diagnosis_summary=self._safe_get_string(response, ["diagnosis_summary", "diagnosis"], "No diagnosis"),
            treatment_plan=self._safe_get_string(response, ["treatment_plan", "treatment"], "Consult doctor"),
            confidence_score=float(response.get("confidence_score", response.get("confidence", 0.5))),
            risk_assessment=self._safe_get_string(response, ["risk_assessment", "risk_level"], RiskLevel.MODERATE),
            reasoning=self._safe_get_string(response, ["reasoning", "reason"], ""),
            escalation_decision=response.get("decision", EscalationDecision.RESOLVE),
            feedback_to_lower_tier=response.get("feedback", None),
            metrics=metrics
        )