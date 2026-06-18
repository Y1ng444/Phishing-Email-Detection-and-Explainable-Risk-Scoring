"""Streamlit app for phishing email detection and explainable risk scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.explainability import get_email_explanation, get_top_features
from src.risk_scoring import DEFAULT_PIPELINE_NAME, load_pipeline, predict_email


MODEL_PATH = Path("models") / DEFAULT_PIPELINE_NAME
METRICS_PATH = Path("results") / "metrics_optimized.csv"


st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="!",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_resources():
    """Load the trained Logistic Regression pipeline."""
    return load_pipeline(DEFAULT_PIPELINE_NAME)


def show_train_first_message() -> None:
    """Show reproducible setup instructions when the model is missing."""
    st.error("Trained pipeline not found.")
    st.code(
        "\n".join(
            [
                "python standardize_datasets.py --input-dir data/raw --output data/processed/phishing_email_standardized.csv",
                "python eda_standardized.py --input data/processed/phishing_email_standardized.csv",
                "python train_optimized.py --input data/processed/phishing_email_standardized.csv",
                "streamlit run app.py",
            ]
        ),
        language="bash",
    )


def risk_color(level: str) -> str:
    """Return a color for the risk level."""
    return {"Low": "#2e7d32", "Medium": "#b26a00", "High": "#c62828"}.get(level, "#444444")


st.title("Phishing Email Detection and Explainable Risk Scoring")
st.caption("Binary text classification using TF-IDF n-grams and Logistic Regression.")

with st.sidebar:
    st.header("Workflow")
    st.markdown(
        """
        1. Standardize raw CSV files.
        2. Run EDA.
        3. Train sklearn pipelines.
        4. Inspect predictions and errors.
        """
    )

    st.header("Expected Model")
    st.code(str(MODEL_PATH), language="text")

    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        logistic = metrics_df[metrics_df["model"] == "logistic_regression"]
        if not logistic.empty:
            row = logistic.iloc[0]
            st.header("Logistic Regression")
            st.metric("Precision", f"{row['precision']:.3f}")
            st.metric("Recall", f"{row['recall']:.3f}")
            st.metric("F1", f"{row['f1']:.3f}")
            st.metric("PR-AUC", f"{row['pr_auc']:.3f}")


if not MODEL_PATH.exists():
    show_train_first_message()
    st.stop()


try:
    pipeline = load_resources()
except Exception as exc:
    st.error(f"Could not load trained pipeline: {exc}")
    show_train_first_message()
    st.stop()


example_email = """From: security@paypa1.example
Subject: Urgent account verification required

We detected unusual activity on your account. Click the link below to verify
your identity within 24 hours or your account will be limited:
http://paypa1-verify.example/login
"""

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

left, right = st.columns([2, 1])

with left:
    email_text = st.text_area(
        "Email text",
        key="email_text",
        height=260,
        placeholder="Paste sender, subject, body, and URLs here...",
    )

with right:
    st.write("")
    st.write("")
    if st.button("Load Example", use_container_width=True):
        st.session_state.email_text = example_email
        st.rerun()
    if st.button("Clear", use_container_width=True):
        st.session_state.email_text = ""
        st.rerun()


if st.button("Analyze Email", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please paste an email before analyzing.")
        st.stop()

    try:
        result = predict_email(email_text, pipeline=pipeline)
        explanation = get_email_explanation(email_text, pipeline, top_n=10)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    st.subheader("Prediction")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Prediction", result["prediction"])
    metric_cols[1].metric("Phishing Score", f"{result['phishing_score']:.3f}")
    metric_cols[2].metric("Risk Score", f"{result['risk_score']}/100")
    metric_cols[3].metric("Risk Level", result["risk_level"])

    st.markdown(
        f"""
        <div style="border-left: 6px solid {risk_color(result['risk_level'])};
                    padding: 0.75rem 1rem; background: #f8f9fa;">
            <strong>{result['risk_level']} risk:</strong> {result['description']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Email Statistics")
    stats_cols = st.columns(3)
    stats_cols[0].metric("Characters", result["email_length"])
    stats_cols[1].metric("Words", result["word_count"])
    stats_cols[2].metric("URLs", result["url_count"])

    st.subheader("Top Active Features")
    active = explanation["top_active_features"]
    if active:
        active_df = pd.DataFrame(active)
        st.dataframe(
            active_df[["feature", "direction", "tfidf_value", "coefficient", "contribution"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No active TF-IDF features from the trained vocabulary were found.")

    st.subheader("Global Model Indicators")
    top_phishing, phishing_coefs, top_ham, ham_coefs = get_top_features(pipeline, n=12)
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Phishing indicators")
        st.dataframe(
            pd.DataFrame({"feature": top_phishing, "coefficient": phishing_coefs}),
            use_container_width=True,
            hide_index=True,
        )
    with col_b:
        st.write("Ham indicators")
        st.dataframe(
            pd.DataFrame({"feature": top_ham, "coefficient": ham_coefs}),
            use_container_width=True,
            hide_index=True,
        )


with st.expander("Project Outputs"):
    st.markdown(
        """
        The optimized workflow writes:

        - `data/processed/phishing_email_standardized.csv`
        - `data/processed/standardization_report.csv`
        - `results/eda_summary.csv`
        - `results/metrics_optimized.csv`
        - `results/logistic_regression_error_analysis.csv`
        - `models/logistic_regression_pipeline.pkl`
        - `models/naive_bayes_pipeline.pkl`
        - `models/linear_svm_pipeline.pkl`
        """
    )
