from typing import List, Dict, Any


class BaseAgent:
    def __init__(self, role: str, tier: int, llm):
        self.role = role
        self.tier = tier
        self.llm = llm

    async def assess(self, case, prior_opinions: List[Dict[str, Any]] = None):
        prior_opinions = prior_opinions or []

        prompt = self._build_prompt(case, prior_opinions)

        response = await self.llm.complete(prompt)

        return {
            "role": self.role,
            "tier": self.tier,
            "risk": self._extract_risk(response.text),
            "confidence": self._extract_confidence(response.text),
            "reasoning": response.text,
            "cost": getattr(response, "cost", 0.0)
        }

    def _build_prompt(self, case, prior_opinions):
        return f"""
You are a {self.role}.

Patient symptoms:
{case.symptoms}

Vitals:
{case.vitals}

Prior expert opinions:
{prior_opinions}

Provide:
- risk level (low, medium, high, critical)
- reasoning
- confidence (0-1)
"""

    def _extract_risk(self, text: str) -> str:
        text = text.lower()
        if "critical" in text:
            return "critical"
        if "high" in text:
            return "high"
        if "medium" in text:
            return "medium"
        return "low"

    def _extract_confidence(self, text: str) -> float:
        # naive fallback
        if "0.9" in text:
            return 0.9
        if "0.8" in text:
            return 0.8
        return 0.7