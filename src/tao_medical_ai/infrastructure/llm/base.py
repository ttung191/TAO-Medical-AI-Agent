from __future__ import annotations

from abc import ABC, abstractmethod

from tao_medical_ai.domain.models import FinalDecisionResponse


class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, decision: FinalDecisionResponse) -> tuple[str, str]:
        raise NotImplementedError
