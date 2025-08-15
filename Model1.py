# Product Review Sentiment + Reasoning
# HuggingFace (sentiment) + Gemini (reasoning/rephrasing) + Areas of Improvement
# ---------------------------------------

import os
import json
import time
import streamlit as st
from transformers import pipeline
import google.generativeai as genai

# -------- BACKEND GEMINI API KEY (set here only) --------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBbERg7SiFvx_AcdgGY_a4ckAxC1jgTlq0")  # set via env or replace locally
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# -------- LOAD SENTIMENT MODEL --------
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_pipe = load_sentiment_pipeline()

def classify_sentiment(text):
    """Return label & score"""
    if not text.strip():
        return "NEUTRAL", 0.0
    result = sentiment_pipe(text[:512])[0]
    return result['label'].upper(), float(result['score'])

def gemini_generate(prompt):
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return (response.text or "").strip()

def generate_reasoning(review_text, sentiment):
    prompt = f"""
The following product review has a {sentiment} sentiment:
"{review_text}"
Explain in 1–2 sentences why this sentiment applies using concrete cues (faulty item, delay, poor UX, etc.).
"""
    return gemini_generate(prompt)

def rephrase_review(review_text):
    prompt = f"""
Rephrase the following review into a neutral, brand-friendly tone suitable for escalation.
Keep it concise (1–2 sentences) and preserve the core issue.
"{review_text}"
"""
    return gemini_generate(prompt)

def generate_improvements(review_text, sentiment):
    """
    Produce 3–5 actionable 'Areas of Improvement' for internal teams (product, CX, ops).
    Short, specific bullet points. Avoid fluff.
    """
    prompt = f"""
You are advising an e-commerce team.

Customer review:
\"\"\"{review_text}\"\"\"

Detected sentiment: {sentiment}

List 3–5 concrete, actionable Areas of Improvement tailored to this feedback.
Write short bullet points that a product/support team can act on immediately.
Avoid generic advice. Be specific to the issues implied by the review.
"""
    text = gemini_generate(prompt)
    # Ensure bullet formatting even if model returns paragraphs
    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    bullets = [f"• {ln}" for ln in lines if ln]
    # Keep between 3 and 5 items if model returned more/less
    if len(bullets) < 3:
        # fallback: split by periods if needed
        parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
        bullets = [f"• {p}" for p in parts[:5]]
    return "\n".join(bullets[:5])

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Review Sentiment + Reasoning", page_icon="🛒", layout="wide")

# -------- CSS for Glass Welcome Page (no background color change) --------
WELCOME_CSS = """
<style>
.welcome-container {
    display: flex; justify-content: center; align-items: center;
    height: 80vh; animation: fadeIn 0.8s ease-in-out;
}
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 3rem; text-align: center; max-width: 620px; width: 92%;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25); animation: slideUp 0.8s ease;
}
.glass-card h1 { font-size: 2rem; font-weight: 800; margin-bottom: 1rem; }
.glass-card p  { font-size: 1.05rem; line-height: 1.55; }
.stButton > button {
    background: #ffffff !important; color: #000000 !important;
    border: 1px solid #d1d5db !important; padding: .9rem 1.3rem !important;
    border-radius: 999px !important; font-weight: 700 !important; font-size: 1rem !important;
    box-shadow: 0 8px 25px rgba(0,0,0,.25) !important;
    transition: transform .08s ease, box-shadow .15s ease !important;
}
.stButton > button:hover { transform: translateY(-1px) scale(1.01); }
.stButton > button:active { transform: translateY(1px) scale(.99); }
@keyframes fadeIn   { from {opacity: 0;} to {opacity: 1;} }
@keyframes slideUp  { from {transform: translateY(24px); opacity: 0;} to {transform: translateY(0); opacity: 1;} }
</style>
"""

# -------- NAV STATE --------
if "view" not in st.session_state:
    st.session_state.view = "start"   # "start" or "analyze"

# -------- WELCOME PAGE --------
def show_start():
    st.markdown(WELCOME_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="welcome-container">
            <div class="glass-card">
                <h1>🛒 Product Review Sentiment + Reasoning</h1>
                <p>
                    Classify reviews (Positive / Neutral / Negative), see a concise “why,”
                    rephrase into a brand-safe tone, and now get <b>Areas of Improvement</b> your team can act on.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    if st.button("Start Analysis"):
        st.session_state.view = "analyze"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------- ANALYZE PAGE --------
def show_analyze():
    st.title("Product Review Sentiment + Reasoning")

    review = st.text_area("Enter Product Review", height=150)

    col1, col2, col3 = st.columns([1,1,1])

    if col1.button("Analyze Review"):
        if review.strip():
            with st.spinner("Analyzing..."):
                label, score = classify_sentiment(review)
                reason = generate_reasoning(review, label)
                improvements = generate_improvements(review, label)
                st.markdown(f"**Sentiment:** {label} (Score: {score:.2f})")
                st.write("**Reason:**", reason)
                st.markdown("**Areas of Improvement:**")
                st.write(improvements)
                # store for export
                st.session_state.last_export = {
                    "review": review,
                    "sentiment": {"label": label, "score": round(score, 4)},
                    "reason": reason,
                    "areas_of_improvement": [ln.lstrip("• ").strip() for ln in improvements.splitlines() if ln.strip()],
                    "timestamp": int(time.time())
                }
        else:
            st.warning("Please enter a review.")

    if col2.button("Rephrase Review"):
        if review.strip():
            with st.spinner("Rephrasing..."):
                rephrased = rephrase_review(review)
                st.write("**Brand-friendly Version:**", rephrased)
                # Attach to export if exists
                if "last_export" in st.session_state:
                    st.session_state.last_export["brand_friendly_rephrase"] = rephrased
        else:
            st.warning("Please enter a review.")

    if col3.button("Export Analysis (JSON)"):
        data = st.session_state.get("last_export")
        # If user presses export without analyze, try to compute minimally
        if not data and review.strip():
            label, score = classify_sentiment(review)
            reason = generate_reasoning(review, label)
            improvements = generate_improvements(review, label)
            data = {
                "review": review,
                "sentiment": {"label": label, "score": round(score, 4)},
                "reason": reason,
                "areas_of_improvement": [ln.lstrip("• ").strip() for ln in improvements.splitlines() if ln.strip()],
                "timestamp": int(time.time())
            }
        if data:
            st.download_button(
                label="Download JSON",
                data=json.dumps(data, indent=2),
                file_name="review_analysis.json",
                mime="application/json"
            )
        else:
            st.warning("Please run an analysis first.")

    st.divider()
    if st.button("⬅️ Back to Start"):
        st.session_state.view = "start"
        st.rerun()

# -------- ROUTER --------
if st.session_state.view == "start":
    show_start()
else:
    show_analyze()
