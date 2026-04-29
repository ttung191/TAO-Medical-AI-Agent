import logging
from collections import defaultdict
from tao_medical_ai.contracts.case import StructuredCase, RoutedAgent
from tao_medical_ai.contracts.enums import EscalationDecision
from tao_medical_ai.agents.dynamic_agent import DynamicMedicalAgent
from tao_medical_ai.orchestration.intra_tier import IntraTierCollaborator

logger = logging.getLogger(__name__)

class TAOOrchestrator:
    def __init__(self):
        self.intra_collab = IntraTierCollaborator()

    async def _mock_recruit_and_route(self, case: StructuredCase):
        team = [RoutedAgent(expertise="Triage Nurse", tier=1), RoutedAgent(expertise="General Practitioner", tier=2)]
        if "tim" in case.chief_complaint.lower() or "ngực" in case.chief_complaint.lower():
            team.append(RoutedAgent(expertise="Cardiologist", tier=3))
        else:
            team.append(RoutedAgent(expertise="Specialist", tier=3))
        return team

    async def process_case(self, case: StructuredCase) -> dict:
        routed_team = await self._mock_recruit_and_route(case)
        
        tier_groups = defaultdict(list)
        for spec in routed_team:
            tier_groups[spec.tier].append(DynamicMedicalAgent(tier=spec.tier, role=spec.expertise))
            
        history = {"prior": [], "prior_notes": ""}
        final_assessment = None
        
        for tier in sorted(tier_groups.keys()):
            agents = tier_groups[tier]
            assessment = await self.intra_collab.run_tier_agents(case, agents, history)
            
            history["prior"].append(assessment)
            history["prior_notes"] += f"[Tier {tier} Consensus]: {assessment.differential[0]}. {assessment.rationale}\n"
            final_assessment = assessment
            
            if assessment.escalation == EscalationDecision.stop:
                break

        return self._format(final_assessment, history, routed_team)

    def _format(self, final, history, team):
        total_prompt = sum(a.token_usage.prompt_tokens for a in history["prior"])
        total_comp = sum(a.token_usage.completion_tokens for a in history["prior"])
        
        return {
            "recruited_team": [{"role": a.expertise, "tier": a.tier} for a in team],
            "final_tier": final.tier,
            "risk": final.risk.value,
            "disposition": final.suggested_disposition.value,
            "diagnosis": final.differential[0],
            "audit_trail": [{"tier": a.tier, "role": a.role, "rationale": a.rationale} for a in history["prior"]],
            "performance_metrics": {
                "total_tokens": total_prompt + total_comp,
                "cost_usd": round(((total_prompt/1e6)*0.075) + ((total_comp/1e6)*0.3), 6)
            }
        }