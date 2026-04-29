import re

class PHIFilter:
    PATTERNS = {
        "phone_vn": r"(0[3|5|7|8|9])+([0-9]{8})",
        "cccd": r"\b[0-9]{12}\b",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "mrn": r"BN\d{6,10}"
    }

    @classmethod
    def redact(cls, text: str) -> str:
        if not text: return text
        sanitized_text = text
        for key, pattern in cls.PATTERNS.items():
            sanitized_text = re.sub(pattern, f"[REDACTED_{key.upper()}]", sanitized_text)
        return sanitized_text