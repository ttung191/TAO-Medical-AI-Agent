import asyncio
from typing import TYPE_CHECKING, List
from tao_medical_ai.contracts.case import StructuredCase, AgentAssessment
from tao_medical_ai.contracts.enums import EscalationDecision

if TYPE_CHECKING:
    from tao_medical_ai.agents.dynamic_agent import DynamicMedicalAgent

class IntraTierCollaborator:
    async def run_tier_agents(self, case: StructuredCase, agents: List['DynamicMedicalAgent'], history: dict) -> AgentAssessment:
        if len(agents) == 1:
            return await agents[0].run(case, history)
            
        tasks = [agent.run(case, history) for agent in agents]
        opinions: List[AgentAssessment] = await asyncio.gather(*tasks)
        
        escalate_ops = [o for o in opinions if o.escalation != EscalationDecision.stop]
        chosen = escalate_ops[0] if escalate_ops else opinions[0]
        
        chosen.rationale = " | ".join([f"({o.role}): {o.rationale}" for o in opinions])
        
        for o in opinions:
            if o != chosen:
                chosen.token_usage.total_tokens += o.token_usage.total_tokens
                chosen.token_usage.prompt_tokens += o.token_usage.prompt_tokens
                chosen.token_usage.completion_tokens += o.token_usage.completion_tokens
                
        return chosen