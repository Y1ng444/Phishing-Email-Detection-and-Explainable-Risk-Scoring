"""Streamlit app for phishing email detection and explainable risk scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.explainability import explain_email_prediction
from src.risk_scoring import load_model, model_exists, score_email


MODEL_PATH = Path("models/phishing_logreg_tfidf.pkl")


st.set_page_config(page_title="Phishing Email Detector", page_icon="!", layout="centered")
st.title("Phishing Email Detection")

if not model_exists(MODEL_PATH):
    st.error("The trained model was not found.")
    st.code(
        "\n".join(
            [
                "python standardize_datasets.py",
                "python eda_standardized.py",
                "python train_optimized.py",
                "streamlit run app.py",
            ]
        ),
        language="bash",
    )
    st.stop()


@st.cache_resource
def get_model():
    """Load the final Logistic Regression pipeline once."""
    return load_model(MODEL_PATH)


model = get_model()

example_email = """Subject: Urgent account verification required

We noticed unusual activity on your account. Please verify your login details
within 24 hours at http://account-verify.example/login to avoid suspension.
"""

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

email_text = st.text_area(
    "Email text",
    key="email_text",
    height=240,
    placeholder="Paste an email subject, body, and URLs here.",
)

col_a, col_b = st.columns(2)
with col_a:
    analyze = st.button("Analyze Email", type="primary", width="stretch")
with col_b:
    if st.button("Load Example", width="stretch"):
        st.session_state.email_text = example_email
        st.rerun()

if analyze:
    if not email_text.strip():
        st.warning("Please paste an email first.")
        st.stop()

    result = score_email(email_text, model)
    explanation = explain_email_prediction(email_text, model, top_n=10)

    st.subheader("Prediction")
    metrics = st.columns(4)
    metrics[0].metric("Label", str(result["prediction"]).title())
    metrics[1].metric("Phishing probability", f"{result['phishing_probability']:.3f}")
    metrics[2].metric("Risk score", f"{result['risk_score']:.1f}/100")
    metrics[3].metric("Risk level", str(result["risk_level"]))

    st.subheader("Top phishing terms in this email")
    if explanation.empty:
        st.info("No positive phishing terms from the trained vocabulary were active.")
    else:
        st.dataframe(
            explanation[["term", "tfidf_value", "coefficient", "contribution"]],
            width="stretch",
            hide_index=True,
        )

metrics_path = Path("results/metrics_optimized.csv")
if metrics_path.exists():
    with st.expander("Model metrics"):
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, width="stretch", hide_index=True)
