"""Streamlit review page, two tabs. Review queue: approve or reject escalated cases through
the API, so an edited draft passes the same checks as an LLM draft. Monitoring: the daily
metrics `make monitor` logs to MLflow. Run with `make review` (port 8502)."""

import httpx
import mlflow
import pandas as pd
import streamlit as st

from app.config import get_settings
from app.llmops.monitor import history
from app.repositories.reason_codes import get_reason_catalog

settings = get_settings()
st.set_page_config(page_title="CreditShield Copilot review", layout="wide")


def api() -> httpx.Client:
    key = settings.api_key.get_secret_value() if settings.api_key else ""
    return httpx.Client(base_url=settings.copilot_api_url, headers={"X-API-Key": key}, timeout=60)


def failed(checks: list[dict]) -> list[dict]:
    return [c for c in checks if not c["passed"]]


def show_flash() -> None:
    kind, text, notice = st.session_state.pop("flash", (None, None, None))
    if kind:
        getattr(st, kind)(text)
    if notice:
        with st.expander("Notice to send", expanded=True):
            st.code(notice, language=None)


def approve_form(item: dict) -> None:
    catalog = get_reason_catalog().features
    draft = item["suggested_draft"]
    with st.form(f"approve-{item['id']}"):
        st.markdown("**Reasons**, one per SHAP feature, in rank order")
        reasons = []
        for i, r in enumerate(draft["reasons"]):
            approved = catalog[r["feature"]].reason if r["feature"] in catalog else "unknown"
            text = st.text_input(
                f"{i + 1}. {r['feature']}  (approved wording: {approved})",
                r["text"],
                key=f"r-{item['id']}-{i}",
            )
            reasons.append({"feature": r["feature"], "text": text})
        paragraph = st.text_area("Reasons paragraph", draft["paragraph"], height=140)
        note = st.text_input("Reviewer note (optional)")
        if st.form_submit_button("Run checks and approve", type="primary"):
            body = {"reasons": reasons, "paragraph": paragraph, "reviewer_note": note or None}
            with api() as c:
                r = c.post(f"/review-queue/{item['id']}/approve", json=body)
            if r.status_code == 200:
                st.session_state["flash"] = (
                    "success", f"Review #{item['id']} approved.", r.json()["notice"]
                )  # fmt: skip
                st.rerun()
            elif r.status_code == 422 and "failed_checks" in str(r.json()):
                st.error("Not approved: the edited draft fails these checks. Fix it or reject.")
                st.table(pd.DataFrame(r.json()["detail"]["failed_checks"]))
            else:
                st.error(f"{r.status_code}: {r.text}")


def reject_form(item: dict) -> None:
    with st.form(f"reject-{item['id']}"):
        note = st.text_input("How it will be handled instead (required)")
        if st.form_submit_button("Reject: handle outside the Copilot"):
            with api() as c:
                r = c.post(f"/review-queue/{item['id']}/reject", json={"reviewer_note": note})
            if r.status_code == 200:
                st.session_state["flash"] = ("info", f"Review #{item['id']} rejected.", None)
                st.rerun()
            st.error(f"{r.status_code}: {r.text}")


def queue_tab() -> None:
    show_flash()
    try:
        with api() as c:
            r = c.get("/review-queue")
        r.raise_for_status()
    except httpx.HTTPError as e:
        st.error(f"Can't reach the API at {settings.copilot_api_url} ({e}). Is `make api` running "
                 "with API_KEY set?")  # fmt: skip
        return
    items = r.json()
    st.metric("Pending reviews", len(items))
    if not items:
        st.success("The queue is empty.")
        return
    labels = {
        f"#{i['id']}  applicant {i['applicant_id']}  {i['decision']}: {i['reason'][:70]}": i
        for i in items
    }
    item = labels[st.selectbox("Case", list(labels))]

    left, right = st.columns([2, 1])
    with right:
        st.markdown(f"**Decision:** {item['decision']}  \n**Queued:** {item['created_at'][:19]}")
        st.markdown(f"**Why it is here:** {item['reason']}")
        if failed(item["check_results"]):
            st.markdown("**Failed checks**")
            st.table(pd.DataFrame(failed(item["check_results"]))[["name", "detail"]])
        if item["trace_id"]:
            st.caption(f"MLflow trace {item['trace_id']}")
    with left:
        features = [r["feature"] for r in (item["suggested_draft"] or {}).get("reasons", [])]
        if item["decision"] != "DENY":
            st.info(f"{item['decision']} is not a denial, so no adverse action notice is sent. "
                    "Reject it and hand it to underwriting.")  # fmt: skip
        elif item["suggested_draft"] is None:
            st.warning("The case data can't be loaded or is invalid, so no notice can be "
                       "drafted here. Reject it and fix the data.")  # fmt: skip
        elif "cnt_children" in features:
            st.warning("A cnt_children reason is never worded by the Copilot, by a person "
                       "either: the protected-attribute check blocks it. Reject it and use a "
                       "compliance-approved letter.")  # fmt: skip
        else:
            approve_form(item)
        reject_form(item)


def monitoring_tab() -> None:
    if not mlflow.is_tracking_uri_set():  # tests point MLflow at a temporary store first
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        h = history()
    except Exception as e:  # noqa: BLE001  MLflow down must not break the review tab
        st.error(f"Can't read MLflow at {settings.mlflow_tracking_uri}: {e}")
        return
    if h.empty:
        st.info("No monitoring runs yet. Run `make monitor` (after `make replay` for traffic).")
        return
    last = h.iloc[-1]
    st.caption(f"Latest monitored day: {h.index[-1]}. Source: `make monitor` runs in MLflow.")
    cols = st.columns(5)
    cols[0].metric("Requests", f"{last['requests']:.0f}")
    cols[1].metric("Escalation rate", f"{last['escalation_rate']:.1%}",
                   f"baseline {last['escalation_baseline']:.1%}", delta_color="off")  # fmt: skip
    cols[2].metric("Cost per notice, p50", f"${last['cost_usd_p50']:.5f}")
    cols[3].metric("Latency p95", f"{last['latency_ms_p95'] / 1000:.1f} s")
    cols[4].metric("Reason-mix PSI", f"{last['reason_psi']:.3f}", "alert at 0.25",
                   delta_color="off")  # fmt: skip
    charts = {
        "Escalation rate vs baseline": ["escalation_rate", "escalation_baseline"],
        "Cost per notice (USD)": ["cost_usd_p50", "cost_usd_p95"],
        "Latency (ms)": ["latency_ms_p50", "latency_ms_p95"],
        "Reason-mix PSI": ["reason_psi"],
    }
    for (title, series), col in zip(charts.items(), st.columns(2) * 2, strict=True):
        with col:
            st.markdown(f"**{title}**")
            st.line_chart(h[[s for s in series if s in h]])
    pass_rates = [c for c in h if c.startswith("pass_rate.")]
    st.markdown("**Check pass rates, latest day**")
    rates = last[pass_rates].rename(lambda c: c.removeprefix("pass_rate."))
    st.dataframe(rates.to_frame("pass rate"))
    alerts = h[h["alerts"] != ""]["alerts"]
    st.markdown("**Alerts**")
    if alerts.empty:
        st.write("None.")
    else:
        st.table(alerts.to_frame())


st.title("CreditShield Copilot")
queue, monitoring = st.tabs(["Review queue", "Monitoring"])
with queue:
    queue_tab()
with monitoring:
    monitoring_tab()
