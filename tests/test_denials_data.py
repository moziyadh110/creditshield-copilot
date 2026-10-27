import pandas as pd
import pytest

from app.config import DATA_DIR
from app.repositories.reason_codes import load_reason_catalog


@pytest.fixture(scope="module")
def denials() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "denials.parquet")


def test_only_allow_listed_columns(denials):
    assert list(denials.columns) == ["applicant_id", "decision", "product", "reasons"]
    for reasons in denials["reasons"]:
        for r in reasons:
            assert set(r) == {"rank", "feature", "impact_score"}


def test_cases_are_denials_or_manual_review(denials):
    assert denials["applicant_id"].is_unique
    assert set(denials["decision"]) == {"DENY", "MANUAL_REVIEW"}
    assert set(denials["product"]) <= {"Cash loans", "Revolving loans"}


def test_reasons_are_ranked_known_features(denials):
    catalog = load_reason_catalog(DATA_DIR / "reason_codes.yaml").features
    for reasons in denials["reasons"]:
        assert 1 <= len(reasons) <= 4
        assert [r["rank"] for r in reasons] == list(range(1, len(reasons) + 1))
        features = [r["feature"] for r in reasons]
        assert len(features) == len(set(features))
        assert all(f in catalog for f in features)
        assert all(r["impact_score"] > 0 for r in reasons)
