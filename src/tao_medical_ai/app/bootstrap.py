from __future__ import annotations

from functools import lru_cache

from tao_medical_ai.app.settings import Settings
from tao_medical_ai.domain.router import DomainRouter
from tao_medical_ai.infrastructure.evidence.retriever import EvidenceRetriever
from tao_medical_ai.infrastructure.llm.provider_factory import ExplainerFactory
from tao_medical_ai.infrastructure.logging.audit import AuditLogger
from tao_medical_ai.infrastructure.privacy.redactor import PHIRedactor
from tao_medical_ai.orchestration.tao_orchestrator import TAOOrchestrator
from tao_medical_ai.safety.policy_engine import PolicyEngine


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def build_orchestrator() -> TAOOrchestrator:
    settings = get_settings()
    return TAOOrchestrator(
        router=DomainRouter(),
        evidence=EvidenceRetriever(),
        explainer=ExplainerFactory.build(settings.llm_provider),
        audit_logger=AuditLogger(settings.trace_dir),
        policy_engine=PolicyEngine(),
        redactor=PHIRedactor(),
    )
