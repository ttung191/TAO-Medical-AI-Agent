from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tao_medical_ai.app.bootstrap import build_orchestrator
from tao_medical_ai.domain.models import (
    CaseRequest,
    DomainName,
    FinalDecisionResponse,
    SpecialtyDescriptor,
)
from tao_medical_ai.domain.specialties import list_specialties
from tao_medical_ai.interfaces.api.auth import enforce_rate_limit, require_api_key
from tao_medical_ai.orchestration.tao_orchestrator import TAOOrchestrator

router = APIRouter(prefix="/cases", tags=["cases"])
meta_router = APIRouter(tags=["specialties"])


def get_orchestrator() -> TAOOrchestrator:
    return build_orchestrator()


@router.post("/analyze", response_model=FinalDecisionResponse, dependencies=[Depends(require_api_key)])
def analyze_case(
    payload: CaseRequest,
    request: Request,
    orchestrator: TAOOrchestrator = Depends(get_orchestrator),
) -> FinalDecisionResponse:
    enforce_rate_limit(request)
    return orchestrator.run(payload)


@meta_router.get("/specialties", response_model=list[SpecialtyDescriptor], dependencies=[Depends(require_api_key)])
def supported_specialties(request: Request) -> list[SpecialtyDescriptor]:
    enforce_rate_limit(request)
    return list_specialties()


@meta_router.get(
    "/intake-templates/{domain}",
    response_model=dict[str, str | list[str]],
    dependencies=[Depends(require_api_key)],
)
def intake_template(domain: DomainName, request: Request, orchestrator: TAOOrchestrator = Depends(get_orchestrator)) -> dict[str, str | list[str]]:
    enforce_rate_limit(request)
    return orchestrator.router.intake_template(domain)
