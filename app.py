"""Streamlit app for phishing email detection and explainable risk scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.explainability import explain_email_prediction, explain_metadata_prediction
from src.risk_scoring import load_model, model_exists, score_email
from src.utils import FINAL_MODEL_PATH


MODEL_PATH = FINAL_MODEL_PATH

EXAMPLE_EMAIL = """Subject: Urgent account verification required

We noticed unusual activity on your account. Please verify your login details
within 24 hours at http://account-verify.example/login to avoid suspension.
"""


def load_example_email() -> None:
    """Load the example email into the text area."""
    st.session_state.email_text = EXAMPLE_EMAIL


st.set_page_config(page_title="Phishing Email Detector", page_icon="!", layout="centered")
st.title("Phishing Email Detection")

if not model_exists(MODEL_PATH):
    st.error(
        "Text + metadata model file not found. Run the full workflow to generate "
        f"`{MODEL_PATH}`."
    )
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
    """Load the final Logistic Regression text+metadata pipeline once."""
    return load_model(MODEL_PATH)


if "email_text" not in st.session_state:
    st.session_state.email_text = ""

model = get_model()

st.button(
    "Load Example",
    width="stretch",
    on_click=load_example_email,
)

email_text = st.text_area(
    "Paste email content here:",
    key="email_text",
    height=250,
)

analyze = st.button("Analyze Email", type="primary", width="stretch")

if analyze:
    if not email_text.strip():
        st.warning("Please paste an email first.")
        st.stop()

    result = score_email(email_text, model)
    explanation = explain_email_prediction(email_text, model, top_n=10)
    metadata_explanation = explain_metadata_prediction(email_text, model, top_n=10)

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

    st.subheader("Metadata indicators")
    if metadata_explanation.empty:
        st.info("No metadata indicators are available for this model.")
    else:
        st.dataframe(
            metadata_explanation[
                ["feature", "raw_value", "coefficient", "contribution", "direction"]
            ],
            width="stretch",
            hide_index=True,
        )

metrics_path = Path("results/metrics_text_metadata.csv")
if metrics_path.exists():
    with st.expander("Model metrics"):
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, width="stretch", hide_index=True)
