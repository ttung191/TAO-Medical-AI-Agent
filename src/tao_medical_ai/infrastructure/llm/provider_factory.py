from __future__ import annotations

from tao_medical_ai.infrastructure.llm.base import BaseExplainer
from tao_medical_ai.infrastructure.llm.offline import OfflineExplainer


class ExplainerFactory:
    @staticmethod
    def build(provider: str) -> BaseExplainer:
        # Hook point for Gemini/OpenAI/Vertex in future revisions.
        if provider == "offline":
            return OfflineExplainer()
        return OfflineExplainer()
