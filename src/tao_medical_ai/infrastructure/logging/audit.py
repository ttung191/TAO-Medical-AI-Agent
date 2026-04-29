from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from tao_medical_ai.domain.models import FinalDecisionResponse


class AuditLogger:
    def __init__(self, trace_dir: str = "runtime") -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.trace_dir / "triage_audit.jsonl"
        self.review_file = self.trace_dir / "human_review_queue.jsonl"

    def next_audit_id(self) -> str:
        return str(uuid4())

    def log_decision(self, decision: FinalDecisionResponse) -> None:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "audit_id": decision.audit_id,
            "case_id": decision.case_id,
            "service_line": decision.service_line,
            "risk": decision.final_risk.value,
            "disposition": decision.final_disposition.value,
            "domain": decision.primary_domain.value,
            "requires_human_review": decision.requires_human_review,
            "governance_flags": decision.governance_flags,
            "complexity_score": decision.complexity_score,
            "tier_path": decision.tier_path,
            "intake_gaps": decision.intake_gaps,
            "recommended_next_steps": decision.recommended_next_steps,
        }
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def enqueue_human_review(self, decision: FinalDecisionResponse) -> None:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "audit_id": decision.audit_id,
            "case_id": decision.case_id,
            "service_line": decision.service_line,
            "risk": decision.final_risk.value,
            "disposition": decision.final_disposition.value,
            "reason": decision.human_review_reason,
            "recommended_next_steps": decision.recommended_next_steps,
        }
        with self.review_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
