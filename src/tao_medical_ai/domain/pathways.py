from __future__ import annotations

from tao_medical_ai.domain.models import Disposition, DomainName, RiskLevel
from tao_medical_ai.domain.specialties import default_disposition

PATHWAY_REGISTRY: dict[DomainName, dict[str, str]] = {
    DomainName.cardiology: {
        "pathway_id": "cardiology-triage-v2",
        "title": "Cardiology chest pain and palpitations triage",
    },
    DomainName.neurology: {
        "pathway_id": "neurology-stroke-triage-v2",
        "title": "Neurology stroke-focused triage",
    },
    DomainName.general_outpatient: {
        "pathway_id": "general-outpatient-assistant-v2",
        "title": "General outpatient assistant",
    },
}

__all__ = ["PATHWAY_REGISTRY", "default_disposition", "Disposition", "RiskLevel", "DomainName"]
