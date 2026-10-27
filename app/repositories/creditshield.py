"""Live CreditShield API (CASE_SOURCE=api): POST /predict and turn the response into a case.

age_years must be in the request (it selects CreditShield's age-group cutoff), but nothing
from the request reaches the LLM: the case keeps only decision, product and reason ranks,
features and impact scores. RC01-RC04 and CreditShield's reason text are dropped.
"""

import httpx

from app.agent.tools import ToolFailure
from app.schemas import DenialCase


class NotADenial(Exception):
    """CreditShield approved the application, so there is no adverse action notice to write."""


class CreditShieldClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout_s,
            transport=transport,
        )

    async def predict(self, applicant_id: int, product: str, application: dict) -> DenialCase:
        try:
            resp = await self._client.post("/predict", json=application)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise ToolFailure(f"CreditShield API timed out: {type(e).__name__}") from e
        except httpx.HTTPError as e:
            raise ToolFailure(f"CreditShield API error: {type(e).__name__}: {e}") from e
        body = resp.json()
        if body["decision"] not in ("DENY", "MANUAL_REVIEW"):
            raise NotADenial(body["decision"])
        reasons = [
            {"rank": r["rank"], "feature": r["feature"], "impact_score": r["impact_score"]}
            for r in body.get("adverse_action_reasons", [])
        ]
        return DenialCase(
            applicant_id=applicant_id, decision=body["decision"], product=product, reasons=reasons
        )

    async def aclose(self) -> None:
        await self._client.aclose()
