from __future__ import annotations

import re

NEGATION_PATTERN = re.compile(r"(?:no|denies|without|not|negative for)\s+$")


def mentions_term(text: str, term: str) -> bool:
    normalized = text.lower()
    for match in re.finditer(re.escape(term.lower().strip()), normalized):
        window = normalized[max(0, match.start() - 24) : match.start()]
        if NEGATION_PATTERN.search(window):
            continue
        return True
    return False


def mentions_any(text: str, terms: list[str] | tuple[str, ...]) -> bool:
    return any(mentions_term(text, term) for term in terms)
