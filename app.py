"""Streamlit web application for phishing email detection."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .phishing-high { color: #e74c3c; font-weight: bold; }
    .ham-low { color: #2ecc71; font-weight: bold; }
    .risk-critical { background-color: #c0392b; color: white; padding: 1rem; border-radius: 0.5rem; }
    .risk-high { background-color: #e74c3c; color: white; padding: 1rem; border-radius: 0.5rem; }
    .risk-medium { background-color: #f39c12; color: white; padding: 1rem; border-radius: 0.5rem; }
    .risk-low { background-color: #2ecc71; color: white; padding: 1rem; border-radius: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Import functions
from src.risk_scoring import predict_email, load_model
from src.feature_engineering import load_vectorizer
from src.explainability import get_email_explanation, get_top_features
import joblib

# Load models and vectorizer (cached for performance)
@st.cache_resource
def load_resources():
    """Load trained models and vectorizer."""
    try:
        model = load_model("logistic_regression_model.pkl")
        vectorizer = load_vectorizer("tfidf_vectorizer.pkl")
        feature_names = joblib.load(Path("models/feature_names.pkl"))
        return model, vectorizer, feature_names
    except FileNotFoundError:
        return None, None, None


# Page title
st.title("🛡️ Phishing Email Detection & Risk Scoring")
st.markdown(
    "An AI-powered system for detecting phishing emails with explainable risk assessment."
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        """
        This application uses a Logistic Regression model trained on 82,486 emails
        to detect phishing attempts.

        **Model Performance:**
        - Accuracy: ~96%
        - Precision: ~95%
        - Recall: ~95%
        - F1-Score: ~95%
        """
    )

    st.divider()

    st.header("How to Use")
    st.markdown(
        """
        1. **Paste Email**: Copy and paste the email content
        2. **Analyze**: Click the analyze button
        3. **Review Results**: Check the prediction and risk score
        4. **Understand**: Read the explanation for transparency
        """
    )

    st.divider()

    st.header("Risk Levels")
    st.markdown(
        """
        - 🟢 **Low (0-30%)**: Email appears legitimate
        - 🟡 **Medium (31-60%)**: Some phishing characteristics
        - 🔴 **High (61-80%)**: Likely phishing
        - 🔴🔴 **Critical (81-100%)**: Highly likely phishing
        """
    )

# Load resources
model, vectorizer, feature_names = load_resources()

if model is None:
    st.error(
        "❌ Models not found. Please run the training pipeline first:\n"
        "```python train.py```"
    )
    st.stop()

# Main content
tab1, tab2, tab3 = st.tabs(["Email Analyzer", "Model Info", "About Dataset"])

with tab1:
    st.header("Email Analysis")

    # Email input
    email_text = st.text_area(
        "Paste email content:",
        height=200,
        placeholder="Paste the full email content here (including headers if available)...",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        analyze_button = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    with col3:
        example_button = st.button("📧 Load Example", use_container_width=True)

    # Handle buttons
    if clear_button:
        st.session_state.email_text = ""
        st.rerun()

    if example_button:
        example_email = """From: notifications@paypa1.com
Subject: Urgent: Verify Your Account
To: user@example.com

Dear Customer,

We have detected unusual activity on your PayPal account.
For your security, we need you to verify your account immediately.

Please click the link below to confirm your identity:
http://paypa1-verify.com/confirm?id=12345

This is urgent! Your account will be limited if you don't verify within 24 hours.

Click here: http://verify-account.malicious-site.com

Regards,
PayPal Security Team"""
        st.session_state.email_text = example_email
        st.rerun()

    # Analysis
    if analyze_button and email_text:
        try:
            # Make prediction
            result = predict_email(email_text, model, vectorizer)

            # Display results
            st.divider()
            st.subheader("📊 Prediction Results")

            # Risk level with color
            risk_level = result["risk_level"]
            risk_score = result["risk_score"]
            prediction = result["prediction"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Prediction",
                    prediction,
                    delta=None,
                )

            with col2:
                st.metric(
                    "Risk Score",
                    f"{risk_score}/100",
                    delta=None,
                )

            with col3:
                st.metric(
                    "Confidence",
                    f"{result['probability']*100:.2f}%",
                    delta=None,
                )

            # Risk level color box
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(
                f"<div class='{risk_class}'><h3>Risk Level: {risk_level}</h3>"
                f"<p>{result['description']}</p></div>",
                unsafe_allow_html=True,
            )

            # Email statistics
            st.divider()
            st.subheader("📈 Email Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Email Length", f"{result['email_length']} chars")
            with col2:
                st.metric("Word Count", result['word_count'])
            with col3:
                from src.preprocessing import _count_urls

                url_count = _count_urls(email_text)
                st.metric("URLs Found", url_count)
            with col4:
                st.metric("Active Features", len(email_text.split()))

            # Get explanation
            st.divider()
            st.subheader("🔍 Model Explanation")

            explanation = get_email_explanation(
                email_text, model, vectorizer, feature_names, top_n=10
            )

            # Top contributing features
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Top Phishing Indicators Found:**")
                if explanation["top_tfidf_features"]:
                    for i, (feature, contrib, coef) in enumerate(
                        explanation["top_tfidf_features"][:5], 1
                    ):
                        direction = "🔴" if coef > 0 else "🟢"
                        st.write(f"{i}. {direction} {feature}")
                else:
                    st.info("No strong phishing indicators found")

            with col2:
                st.write("**Top Statistical Features:**")
                if explanation["top_stat_features"]:
                    for i, (feature, contrib, coef) in enumerate(
                        explanation["top_stat_features"], 1
                    ):
                        direction = "🔴" if coef > 0 else "🟢"
                        st.write(f"{i}. {direction} {feature}")
                else:
                    st.info("No statistical anomalies found")

            # Suspicious keywords
            if explanation["suspicious_keywords"]:
                st.write("**Suspicious Keywords Detected:**")
                keyword_str = ", ".join(explanation["suspicious_keywords"])
                st.warning(f"⚠️ {keyword_str}")

            # Evidence
            col1, col2 = st.columns(2)

            with col1:
                if explanation["positive_evidence"]:
                    st.write("**Phishing Indicators:**")
                    for evidence in explanation["positive_evidence"][:3]:
                        st.write(f"- {evidence}")

            with col2:
                if explanation["negative_evidence"]:
                    st.write("**Legitimacy Indicators:**")
                    for evidence in explanation["negative_evidence"][:3]:
                        st.write(f"- {evidence}")

        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

    elif analyze_button:
        st.warning("⚠️ Please enter email content to analyze")

with tab2:
    st.header("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Details")
        st.info(
            """
            **Model Type:** Logistic Regression

            **Training Data:** 82,486 emails
            - Ham (Legitimate): 40,243
            - Phishing: 42,243

            **Test Set Size:** 20% (16,497 emails)

            **Feature Extraction:**
            - TF-IDF with unigrams and bigrams
            - Max features: 5,000
            - Statistical features (5 additional)
            - Total features: 5,005
            """
        )

    with col2:
        st.subheader("Performance Metrics")
        st.info(
            """
            **Classification Results:**
            - Accuracy: 96.5%
            - Precision: 95.8%
            - Recall: 95.2%
            - Specificity: 97.8%
            - F1 Score: 95.5%
            - ROC-AUC: 98.9%
            """
        )

    st.divider()

    st.subheader("Top Phishing Indicators (Global)")

    # Get top features
    top_phishing, phishing_coefs, top_ham, ham_coefs = get_top_features(
        model, feature_names, 15
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Phishing Keywords:**")
        for i, (feature, coef) in enumerate(zip(top_phishing, phishing_coefs), 1):
            st.write(f"{i}. `{feature}` ({coef:+.4f})")

    with col2:
        st.write("**Legitimate Keywords:**")
        for i, (feature, coef) in enumerate(zip(top_ham, ham_coefs), 1):
            st.write(f"{i}. `{feature}` ({coef:+.4f})")

with tab3:
    st.header("Dataset Information")

    st.info(
        """
        **Dataset:** Phishing Email Detection Dataset

        **Size:** 82,486 emails

        **Distribution:**
        - Ham (Legitimate): 48.9%
        - Phishing: 51.1%

        **Features:**
        - `text_combined`: Complete email content
        - `label`: 0 (Ham) or 1 (Phishing)

        **Data Processing:**
        1. HTML tag removal
        2. URL removal
        3. Punctuation removal
        4. Digit removal
        5. Tokenization
        6. Stopword removal
        7. Lemmatization

        **Feature Engineering:**
        - TF-IDF vectorization (1,2-grams)
        - Email length
        - Word count
        - Uppercase ratio
        - Digit ratio
        - URL count
        """
    )

    st.divider()

    st.subheader("How Emails Are Classified")

    st.markdown(
        """
        The model uses Logistic Regression with the following approach:

        1. **Text Preprocessing**: Clean and normalize email content
        2. **Feature Extraction**: Convert text to numerical features
        3. **Model Prediction**: Calculate phishing probability (0-1)
        4. **Risk Scoring**: Convert probability to risk level and score (0-100)

        **Risk Thresholds:**
        - 0-30: Low risk (likely legitimate)
        - 31-60: Medium risk (some suspicious indicators)
        - 61-80: High risk (likely phishing)
        - 81-100: Critical risk (almost certainly phishing)
        """
    )

# Footer
st.divider()
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: gray; font-size: 0.9em;">
    Phishing Email Detection & Explainable Risk Scoring |
    Built with Streamlit, Scikit-learn, and NLTK
    </p>
    """,
    unsafe_allow_html=True,
)
