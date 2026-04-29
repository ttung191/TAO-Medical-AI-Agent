from __future__ import annotations

from collections import defaultdict, deque
from time import time

from fastapi import Header, HTTPException, Request, status

from tao_medical_ai.app.bootstrap import get_settings

_REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.enable_auth:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def enforce_rate_limit(request: Request) -> None:
    settings = get_settings()
    if not settings.enable_rate_limit:
        return
    client = request.client.host if request.client else "unknown"
    now = time()
    bucket = _REQUEST_LOG[client]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    bucket.append(now)
