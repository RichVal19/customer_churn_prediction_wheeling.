"""
Customer Churn Risk Predictor  -  Streamlit app
Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then point Streamlit Community Cloud at app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import churn_core as cc

APP_DIR = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Page setup + light styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {"LOW": "#16a34a", "MEDIUM": "#f59e0b", "HIGH": "#dc2626"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      .hero {
        padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #1e3a8a 0%, #4f46e5 55%, #7c3aed 100%);
        color: #fff;
      }
      .hero h1 {margin: 0 0 .25rem 0; font-size: 1.9rem; color: #fff;}
      .hero p  {margin: 0; opacity: .9; font-size: 1rem;}
      .badge {
        display: inline-block; padding: .35rem 1rem; border-radius: 999px;
        font-weight: 700; letter-spacing: .06em; color: #fff; font-size: 1.05rem;
      }
      .tip {
        padding: .6rem .9rem; margin: .35rem 0; border-radius: 10px;
        border-left: 4px solid #4f46e5; background: rgba(79,70,229,.08);
      }
    </style>
    <div class="hero">
      <h1>📉 Customer Churn Risk Predictor</h1>
      <p>Describe a customer and get a churn probability, a risk level, and ideas to keep them.
         Powered by a Random Forest trained on the Telco Customer Churn dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def _load_from_disk(path_str: str, mtime: float):
    return cc.load_model_from_path(path_str)


@st.cache_resource(show_spinner="Loading uploaded model…")
def _load_from_upload(raw: bytes):
    return cc.load_model_from_bytes(raw)


def get_model():
    """Priority: uploaded file > MODEL_PATH env var / secret > auto-detected file."""
    uploaded = st.session_state.get("uploaded_model")
    if uploaded is not None:
        return _load_from_upload(uploaded.getvalue()), f"uploaded: {uploaded.name}"

    configured = os.environ.get("MODEL_PATH")
    try:
        configured = configured or st.secrets.get("MODEL_PATH")
    except Exception:  # no secrets file locally
        pass
    path = (APP_DIR / configured) if configured else cc.find_model_file(APP_DIR)
    if path and Path(path).is_file():
        return _load_from_disk(str(path), Path(path).stat().st_mtime), Path(path).name
    return None, None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Model")
    st.file_uploader(
        "Use a different model file",
        type=["pkl", "joblib"],
        key="uploaded_model",
        help="Optional. By default the app loads the model file stored next to app.py.",
    )

    st.subheader("Risk bands")
    medium_thr = st.slider("MEDIUM risk from", 0.05, 0.90, cc.DEFAULT_MEDIUM_THRESHOLD, 0.05,
                           help="Notebook default: 0.30")
    high_thr = st.slider("HIGH risk from", 0.10, 0.95, cc.DEFAULT_HIGH_THRESHOLD, 0.05,
                         help="Notebook default: 0.50")
    if medium_thr >= high_thr:
        st.warning("MEDIUM must be lower than HIGH. Using notebook defaults instead.")
        medium_thr, high_thr = cc.DEFAULT_MEDIUM_THRESHOLD, cc.DEFAULT_HIGH_THRESHOLD
    st.caption(
        "Lower thresholds catch more churners (higher recall) but flag more customers "
        "who were not leaving (lower precision). It is a business decision."
    )

model, model_label = get_model()

with st.sidebar:
    st.divider()
    if model is not None:
        st.success(f"Model loaded: {model_label}")
    else:
        st.error("No model file found")

if model is None:
    st.error(
        "**No trained model found.** Put your saved model (for example `model.pkl` or "
        "`churn_model.joblib`) in the same folder as `app.py`, or upload it from the sidebar."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Session-state defaults + quick scenarios (the three examples from the notebook)
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    tenure=12, monthly=70.0, contract="Month-to-month", internet="Fiber optic",
    payment="Electronic check", paperless="Yes", tech="No", security="No", senior=False,
)
PRESETS = {
    "Custom": None,
    "New customer, flexible contract": dict(
        tenure=2, monthly=95.0, contract="Month-to-month", internet="Fiber optic",
        payment="Electronic check", paperless="Yes", tech="No", security="No", senior=False),
    "Loyal long-term customer": dict(
        tenure=60, monthly=60.0, contract="Two year", internet="DSL",
        payment="Credit card (automatic)", paperless="No", tech="Yes", security="Yes", senior=False),
    "Customer in between": dict(
        tenure=24, monthly=75.0, contract="One year", internet="DSL",
        payment="Mailed check", paperless="No", tech="No", security="No", senior=False),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("preset", "Custom")


def apply_preset():
    p = PRESETS[st.session_state.preset]
    if p:
        st.session_state.update(p)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_single, tab_batch, tab_model = st.tabs(["🔮 Single customer", "📂 Score a customer list", "🧠 About the model"])

# =========================================================================== #
# TAB 1  -  single customer
# =========================================================================== #
with tab_single:
    left, right = st.columns([1, 1.25], gap="large")

    # ---------------------------- inputs ---------------------------------- #
    with left:
        st.subheader("Customer profile")
        st.selectbox("Quick scenario", list(PRESETS), key="preset", on_change=apply_preset)

        c1, c2 = st.columns(2)
        c1.slider("Tenure (months)", 0, 72, key="tenure")
        c2.slider("Monthly charges ($)", 18.0, 120.0, step=0.5, key="monthly")

        st.selectbox("Contract", cc.CATEGORY_OPTIONS["Contract"], key="contract")
        st.selectbox("Internet service", cc.CATEGORY_OPTIONS["InternetService"], key="internet")

        # Keep dependent fields consistent: no internet -> no tech support / security.
        no_net = st.session_state.internet == "No"
        for k in ("tech", "security"):
            if no_net:
                st.session_state[k] = "No internet service"
            elif st.session_state[k] == "No internet service":
                st.session_state[k] = "No"
        addon_options = ["No internet service"] if no_net else ["No", "Yes"]

        c3, c4 = st.columns(2)
        c3.selectbox("Tech support", addon_options, key="tech", disabled=no_net)
        c4.selectbox("Online security", addon_options, key="security", disabled=no_net)

        st.selectbox("Payment method", cc.CATEGORY_OPTIONS["PaymentMethod"], key="payment")
        c5, c6 = st.columns(2)
        c5.selectbox("Paperless billing", cc.CATEGORY_OPTIONS["PaperlessBilling"], key="paperless")
        c6.checkbox("Senior citizen", key="senior")
        st.caption("Total charges are estimated as tenure × monthly charges, as in the notebook.")

    customer = dict(
        tenure=int(st.session_state.tenure),
        MonthlyCharges=float(st.session_state.monthly),
        TotalCharges=float(st.session_state.tenure) * float(st.session_state.monthly),
        SeniorCitizen=int(st.session_state.senior),
        Contract=st.session_state.contract,
        InternetService=st.session_state.internet,
        PaymentMethod=st.session_state.payment,
        PaperlessBilling=st.session_state.paperless,
        TechSupport=st.session_state.tech,
        OnlineSecurity=st.session_state.security,
    )

    try:
        prob = float(cc.predict_proba(model, cc.single_customer_frame(**customer))[0])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        st.stop()
    level = cc.risk_level(prob, medium_thr, high_thr)
    color = COLORS[level]

    # ---------------------------- results --------------------------------- #
    with right:
        st.subheader("Churn risk")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 52}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.28},
                "steps": [
                    {"range": [0, medium_thr * 100], "color": "rgba(22,163,74,.22)"},
                    {"range": [medium_thr * 100, high_thr * 100], "color": "rgba(245,158,11,.25)"},
                    {"range": [high_thr * 100, 100], "color": "rgba(220,38,38,.22)"},
                ],
            },
        ))
        gauge.update_layout(height=270, margin=dict(l=25, r=25, t=25, b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge)

        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<span class="badge" style="background:{color}">{level} RISK</span>', unsafe_allow_html=True)
        m2.metric("Churn probability", f"{prob * 100:.1f}%")
        m3.metric("vs. average customer", f"{(prob - cc.BASE_CHURN_RATE) * 100:+.1f} pts",
                  delta=f"{prob / cc.BASE_CHURN_RATE:.1f}× baseline", delta_color="off",
                  help=f"Baseline: about {cc.BASE_CHURN_RATE * 100:.1f}% of Telco customers churn.")

        st.markdown("##### 💡 Suggested actions")
        for tip in cc.recommendations(customer, level):
            st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)

    # ---------------------------- what-if --------------------------------- #
    st.divider()
    w1, w2 = st.columns(2, gap="large")

    with w1:
        st.markdown("##### 🔧 What would lower this risk?")
        st.caption("Each row changes one thing and keeps everything else the same.")
        levers = cc.what_if_levers(model, customer)
        if levers.empty:
            st.info("No single change lowers the risk further for this customer.")
        else:
            st.dataframe(levers, hide_index=True)

    with w2:
        st.markdown("##### 📑 Same customer, different contract")
        comp = cc.contract_comparison(model, customer)
        bar = go.Figure(go.Bar(
            x=comp["Contract"], y=comp["Churn risk %"],
            marker_color=["#4f46e5" if c == customer["Contract"] else "rgba(79,70,229,.35)" for c in comp["Contract"]],
            text=comp["Churn risk %"].map(lambda v: f"{v:.1f}%"), textposition="outside",
        ))
        bar.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(range=[0, 105], title="Churn risk %"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar)

# =========================================================================== #
# TAB 2  -  batch scoring
# =========================================================================== #
with tab_batch:
    st.subheader("Score many customers at once")
    st.write(
        "Upload a CSV with one customer per row. Required columns: "
        f"`{', '.join(c for c in cc.FEATURES if c != 'TotalCharges')}`. "
        "`TotalCharges` is optional (blank values are estimated as tenure × monthly charges)."
    )
    st.download_button(
        "⬇️ Download CSV template",
        cc.sample_batch_template().to_csv(index=False).encode(),
        file_name="churn_template.csv", mime="text/csv",
    )
    csv_file = st.file_uploader("Customer CSV", type=["csv"], key="batch_csv")

    if csv_file is not None:
        try:
            raw_df = pd.read_csv(csv_file)
        except Exception as exc:  # noqa: BLE001
            raw_df = None
            st.error(f"Could not read the CSV: {exc}")

        required = [c for c in cc.FEATURES if c != "TotalCharges"]
        missing = [c for c in required if raw_df is not None and c not in raw_df.columns]
        if raw_df is None:
            pass
        elif missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            work = cc.clean_raw(raw_df)
            # warn about category values the model has never seen
            for col, allowed in cc.CATEGORY_OPTIONS.items():
                unknown = sorted(set(work[col].dropna().astype(str)) - set(allowed))
                if unknown:
                    st.warning(f"`{col}` has unexpected values {unknown}. They will be treated as 'none of the known categories'.")

            probs = cc.predict_proba(model, work)
            result = raw_df.copy()
            result["ChurnRisk%"] = (probs * 100).round(1)
            result["RiskLevel"] = cc.risk_levels(probs, medium_thr, high_thr)
            result = result.sort_values("ChurnRisk%", ascending=False).reset_index(drop=True)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Customers scored", f"{len(result):,}")
            k2.metric("HIGH risk", f"{(result.RiskLevel == 'HIGH').sum():,}")
            k3.metric("MEDIUM risk", f"{(result.RiskLevel == 'MEDIUM').sum():,}")
            k4.metric("Average risk", f"{result['ChurnRisk%'].mean():.1f}%")

            hist = go.Figure(go.Histogram(x=result["ChurnRisk%"], nbinsx=20, marker_color="#4f46e5"))
            hist.add_vline(x=medium_thr * 100, line_dash="dash", line_color=COLORS["MEDIUM"])
            hist.add_vline(x=high_thr * 100, line_dash="dash", line_color=COLORS["HIGH"])
            hist.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Churn risk %", yaxis_title="Customers",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(hist)

            st.markdown("##### Retention list (highest risk first)")
            st.dataframe(
                result,
                hide_index=True,
                column_config={
                    "ChurnRisk%": st.column_config.ProgressColumn(
                        "ChurnRisk%", min_value=0, max_value=100, format="%.1f%%"),
                },
            )
            st.download_button(
                "⬇️ Download scored list",
                result.to_csv(index=False).encode(),
                file_name="churn_scored_customers.csv", mime="text/csv",
            )

# =========================================================================== #
# TAB 3  -  about the model
# =========================================================================== #
with tab_model:
    a, b = st.columns([1.2, 1], gap="large")

    with a:
        st.subheader("What drives the predictions?")
        imps = cc.feature_importance(model, top_n=10)
        if imps is None:
            st.info("This model does not expose feature importances.")
        else:
            fig = go.Figure(go.Bar(x=imps.values, y=imps.index, orientation="h", marker_color="#7c3aed"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Importance score",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)
            st.caption("Top 10 features by Random Forest importance. Importance shows what the model "
                       "relies on, not what causes churn.")

    with b:
        st.subheader("Model facts")
        est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
        facts = {"Type": type(est).__name__}
        if hasattr(est, "n_estimators"):
            facts["Trees"] = est.n_estimators
        facts["Input features"] = len(cc.expected_columns(model))
        facts["Risk bands"] = f"LOW < {medium_thr:.2f} ≤ MEDIUM < {high_thr:.2f} ≤ HIGH"
        st.table(pd.Series(facts, name="Value").astype(str))

    st.divider()
    st.subheader("How to read the results")
    st.markdown(
        """
- **The probability is a score, not a certainty.** "72%" means customers like this one leave much more often than average.
- **Accuracy alone can fool you.** Only about a quarter of customers churn, so a model that always says "stays" is already ~73% accurate and catches nobody. Judge the model on **recall** (how many churners it finds) and **precision** (how often an alert is right).
- **The threshold is a business choice.** Retention calls are cheap and losing a customer is expensive? Lower the thresholds in the sidebar. Each offer is costly? Raise them.
- **Correlation is not causation.** The *What would lower this risk?* table shows how the model's score changes, not a guaranteed effect of an offer. Test retention actions before rolling them out.
        """
    )
    with st.expander("Features used by the model"):
        st.write("**Numeric:** " + ", ".join(cc.NUMERIC_FEATURES))
        st.write("**Categorical (one-hot encoded):** " + ", ".join(cc.CATEGORICAL_FEATURES))
