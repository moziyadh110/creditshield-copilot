import pytest

from app.config import DATA_DIR
from app.repositories.reason_codes import ReasonCatalog, load_reason_catalog

# CreditShield v1.1.0 FEATURE_COLS (fair-lend-ml models/train_baseline.py), copied as data.
CREDITSHIELD_FEATURES = [
    "years_employed", "amt_income_total", "amt_credit", "amt_annuity",
    "credit_income_ratio", "annuity_income_ratio", "goods_to_credit_ratio",
    "owns_car", "owns_realty", "cnt_children",
    "bureau_loan_count", "active_loans", "closed_loans", "avg_days_overdue",
    "max_days_overdue", "bureau_debt_ratio", "total_bureau_credit",
    "prev_app_count", "prev_approved", "prev_refused", "credit_application_ratio",
    "late_payments", "avg_days_late", "payment_volatility", "total_installments",
    "avg_cc_utilization", "max_cc_utilization", "avg_days_past_due_cc", "num_credit_cards",
    "pos_late_payments", "avg_pos_days_past_due",
    "exposure_at_risk", "late_payment_rate", "high_util_x_high_leverage",
]  # fmt: skip


@pytest.fixture(scope="module")
def catalog() -> ReasonCatalog:
    return load_reason_catalog(DATA_DIR / "reason_codes.yaml")


def test_covers_exactly_the_model_features(catalog):
    assert len(CREDITSHIELD_FEATURES) == 34
    assert list(catalog.features) == CREDITSHIELD_FEATURES


def test_age_is_never_a_reason(catalog):
    assert "age_years" not in catalog.features
    for rc in catalog.features.values():
        assert "age" not in (rc.reason or "").lower().split()


def test_children_always_escalate(catalog):
    rc = catalog.features["cnt_children"]
    assert rc.escalate and rc.sensitive and rc.reason is None
    assert [f for f, rc in catalog.features.items() if rc.escalate] == ["cnt_children"]


def test_employment_is_worded_as_length_of_employment(catalog):
    rc = catalog.features["years_employed"]
    assert rc.sensitive and not rc.escalate
    assert rc.reason == "Length of employment"


def test_codes_are_stable_and_in_feature_order(catalog):
    assert [rc.code for rc in catalog.features.values()] == [f"CS{i:02d}" for i in range(1, 35)]


def test_no_reason_uses_its_own_banned_wording(catalog):
    for feature, rc in catalog.features.items():
        for phrase in rc.banned:
            assert phrase.lower() not in (rc.reason or "").lower(), (feature, phrase)


def test_duplicate_codes_are_rejected():
    entry = {"code": "CS01", "source": "application", "reason": "x", "banned": []}
    entry |= {"escalate": False, "sensitive": False}
    with pytest.raises(ValueError, match="unique"):
        ReasonCatalog.model_validate({"version": 1, "features": {"a": entry, "b": entry}})
