"""Denial cases. CASE_SOURCE=file (default) reads data/denials.parquet; no CreditShield needed."""

from functools import lru_cache

import pandas as pd

from app.config import get_settings
from app.schemas import DenialCase


class CaseNotFound(LookupError):
    pass


@lru_cache
def load_denials() -> dict[int, DenialCase]:
    df = pd.read_parquet(get_settings().denials_path)
    return {
        int(row.applicant_id): DenialCase(
            applicant_id=int(row.applicant_id),
            decision=row.decision,
            product=row.product,
            reasons=list(row.reasons),
        )
        for row in df.itertuples(index=False)
    }


def get_case(applicant_id: int) -> DenialCase:
    if get_settings().case_source != "file":
        raise NotImplementedError("CASE_SOURCE=api (live CreditShield) is not built yet")
    try:
        return load_denials()[applicant_id]
    except KeyError:
        raise CaseNotFound(f"applicant {applicant_id} is not in denials.parquet") from None
