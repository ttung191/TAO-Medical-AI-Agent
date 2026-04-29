from __future__ import annotations

import re
from typing import Any

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s]{7,}\d)")
DOB_RE = re.compile(r"\b(?:dob|date of birth)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)
MRN_RE = re.compile(r"\b(?:mrn|medical record number)[:\s#-]*[A-Za-z0-9-]{4,}\b", re.IGNORECASE)


class PHIRedactor:
    def redact_text(self, text: str) -> str:
        text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        text = PHONE_RE.sub("[REDACTED_PHONE]", text)
        text = DOB_RE.sub("[REDACTED_DOB]", text)
        text = MRN_RE.sub("[REDACTED_MRN]", text)
        return text

    def redact_object(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self.redact_object(item) for item in value]
        if isinstance(value, dict):
            return {key: self.redact_object(item) for key, item in value.items()}
        return value
