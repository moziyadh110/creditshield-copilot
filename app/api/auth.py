import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """CreditShield's pattern: fail closed when no key is configured, and compare in
    constant time (`!=` returns at the first differing character, so response timing could
    leak how much of a guessed key was right)."""
    expected = get_settings().api_key
    if expected is None:
        raise HTTPException(status_code=503, detail="API_KEY is not configured on the server")
    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
