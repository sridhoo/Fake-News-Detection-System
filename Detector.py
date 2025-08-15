# streamlit run app.py
import os, json, textwrap, re
from typing import List, Dict

import streamlit as st


st.set_page_config(page_title="Fake News Detector", layout="centered")

# ---- STATE ----
if "started" not in st.session_state:
    st.session_state.started = False

# ---- STYLES ----
st.markdown("""
<style>
@keyframes floatin { from { transform: translateY(12px); opacity:0 } to { transform: translateY(0); opacity:1 } }
.glass {
  animation: floatin .6s ease-out;
  backdrop-filter: blur(12px);
  background: rgba(255,255,255,.35);
  border: 1px solid rgba(255,255,255,.45);
  border-radius: 16px; padding: 26px 26px; box-shadow: 0 12px 32px rgba(0,0,0,.06);
}
.badge {
  display:inline-block; padding:8px 12px; border-radius:999px; color:#fff; font-weight:600;
}
.smallmuted { opacity:.65; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# =========================
# PAGE 1 — WELCOME (GLASS)
# =========================
if not st.session_state.started:
    st.title("Fake News Detection with Explanation")
    st.markdown("""
<div class="glass">
  <h3 style="margin-top:0">Welcome</h3>
  <p>This mini app takes any news article (paste text or give a URL) and instantly labels it
  <b>Fake</b> or <b>Real</b> using a zero-shot model (no training needed). It also retrieves a few short snippets
  (Wikipedia + web) to make the decision more transparent. Finally, it generates a concise explanation using <b>Gemini</b>
  which you can copy into your moderation notes.</p>
  <p class="smallmuted">Tip: keep inputs under a few thousand characters for speed.</p>
</div>
""", unsafe_allow_html=True)

    st.write("")
    if st.button("🚀 Start the analysis"):
        st.session_state.started = True
        st.rerun()           # jump to analysis page
    st.stop()                # IMPORTANT: keep inside the welcome branch

# =========================
# PAGE 2 — ANALYSIS
# =========================
st.title("Analysis")
st.caption("Paste text or fetch by URL → classify → view snippets → generate & copy reasoning")

# ---- Lazy imports so the welcome page stays snappy ----
from transformers import pipeline
import wikipedia
from duckduckgo_search import DDGS
from newspaper import Article
import google.generativeai as genai

# ---- Keys / Gemini ----
import google.generativeai as genai

# Directly hardcode your key here (not recommended for production)
GEMINI_API_KEY = "AIzaSyBbERg7SiFvx_AcdgGY_a4ckAxC1jgTlq0"  # your real key
genai.configure(api_key=GEMINI_API_KEY)

# ---- Classifier (zero-shot) ----
@st.cache_resource(show_spinner=False)
def load_zero_shot():
    # MNLI zero-shot model (no dataset needed)
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

clf = load_zero_shot()
LABELS = ["fake news", "real news"]

# ---- Utils ----
def extract_text(url: str) -> str:
    art = Article(url)
    art.download(); art.parse()
    return art.text or ""

def clean_preview(x: str, n=700):
    x = re.sub(r"\s+", " ", x).strip()
    return x[:n] + ("..." if len(x) > n else "")

def best_similar(query: str, candidates: List[str], k=2) -> List[int]:
    if not candidates: return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(max_features=4000).fit([query] + candidates)
        qv = vec.transform([query]); cv = vec.transform(candidates)
        sims = cosine_similarity(qv, cv).ravel()
        return sims.argsort()[::-1][:k].tolist()
    except Exception:
        return list(range(min(k, len(candidates))))

def fetch_wikipedia(query: str, k=2) -> List[Dict]:
    out=[]
    try:
        hits = wikipedia.search(query, results=6)
        pages=[]
        for t in hits:
            try:
                p = wikipedia.page(t, auto_suggest=False)
                pages.append({"title": p.title, "url": p.url, "summary": p.summary[:1200]})
            except Exception:
                continue
        if not pages: return []
        idx = best_similar(query, [p["summary"] for p in pages], k=min(k, len(pages)))
        return [pages[i] for i in idx]
    except Exception:
        return out

def fetch_web(query: str, k=1) -> List[Dict]:
    out=[]
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=6):
                if "title" in r and "href" in r and "body" in r:
                    out.append({"title": r["title"], "url": r["href"], "summary": r["body"][:400]})
                if len(out) >= k: break
    except Exception:
        pass
    return out

def format_sources(sources: List[Dict]) -> str:
    if not sources: return "No strong corroboration found."
    lines=[]
    for s in sources[:3]:
        t=s.get("title","Source"); u=s.get("url",""); sm=s.get("summary","")
        lines.append(f"- {t} — {u}\n  {sm}")
    return "\n".join(lines)

def generate_reasoning_with_gemini(article: str, label: str, prob: float, sources: List[Dict]) -> str:
    prompt = f"""
You are an AI content-moderation assistant. Given a news passage, a model's label and probability,
and a few retrieved source snippets, write a concise, neutral explanation of WHY the article appears {label.upper()}
(or why it might be reliable). Cite snippets qualitatively (no URLs in prose). Avoid sensational language.
Maximum 120 words.

Label: {label} (p={prob:.2f})

Retrieved snippets:
{format_sources(sources)}

Article (first 1000 chars):
{article[:1000]}
"""
    if GEMINI_API_KEY:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    # fallback (no key)
    stance = "likely FAKE" if label.lower()=="fake" else "likely REAL"
    return textwrap.dedent(f"""
    The article is **{stance}** with model confidence {prob:.2f}. We compared core claims with a few general sources.
    When core assertions lack corroboration or conflict with multiple references, we lean Fake; consistent reporting across
    independent sources suggests Real.

    Supporting snippets:
    {format_sources(sources)}
    """).strip()

# ---- Sidebar ----
with st.sidebar:
    mode = st.radio("Input", ["Paste text", "Fetch from URL"], horizontal=True)
    st.markdown("---")
    if "last" in st.session_state:
        if st.button("Export last result (.json)"):
            st.download_button(
                "Download JSON",
                data=json.dumps(st.session_state["last"], indent=2),
                file_name="fake_news_result.json",
                mime="application/json",
                key="dlbtn_json",
                use_container_width=True
            )

# ---- Input area ----
article = ""
url = ""

if mode == "Paste text":
    article = st.text_area("Paste the news article or claim", height=220, placeholder="Paste article text here…")
else:
    url = st.text_input("Article URL", placeholder="https://example.com/news/story")
    if st.button("Fetch article"):
        if url.strip():
            with st.spinner("Fetching article…"):
                try:
                    article = extract_text(url.strip())
                    st.success("Fetched! You can edit below before analyzing.")
                except Exception as e:
                    st.error(f"Could not fetch: {e}")
        else:
            st.warning("Please enter a URL.")
    article = st.text_area("Article text", value=article, height=220)

colA, colB, colC = st.columns([1,1,1])
with colA:
    classify_btn = st.button("Classify")
with colB:
    gen_btn = st.button("Generate reasoning (Gemini)")
with colC:
    copy_btn = st.button("Copy reasoning")

# ---- Classification ----
if classify_btn and article.strip():
    with st.spinner("Classifying and retrieving supporting snippets…"):
        res = clf(article, LABELS, multi_label=False)
        scores = {res["labels"][i]: float(res["scores"][i]) for i in range(len(res["labels"]))}
        p_fake = scores.get("fake news", 0.5)
        p_real = scores.get("real news", 0.5)
        label = "Fake" if p_fake >= p_real else "Real"
        prob = p_fake if label == "Fake" else p_real

        # retrieval
        query = article[:300]
        sources = fetch_wikipedia(query, k=2) + fetch_web(query, k=1)

    st.subheader("Classification result")
    color = "#c62828" if label == "Fake" else "#2e7d32"
    st.markdown(
        f"<span class='badge' style='background:{color};'>{label} — confidence {prob:.2f}</span>",
        unsafe_allow_html=True
    )

    st.subheader("Snippet of retrieved supporting sources")
    if not sources:
        st.info("No strong snippets found. Try a more specific paragraph or include proper nouns.")
    else:
        for s in sources:
            st.markdown(f"**[{s.get('title','Source')}]({s.get('url','#')})**")
            st.write(s.get('summary',''))

    # save for later actions
    st.session_state["last"] = {
        "prediction": label,
        "confidence": round(prob, 3),
        "sources": sources,
        "article_preview": clean_preview(article, 600),
        "reasoning": ""
    }

elif classify_btn:
    st.warning("Please paste text or fetch a URL first.")

# ---- Reasoning ----
if gen_btn:
    if "last" not in st.session_state or st.session_state["last"].get("prediction") is None:
        st.warning("Run classification first.")
    else:
        data = st.session_state["last"]
        article_preview = data["article_preview"]
        label = data["prediction"]
        prob = data["confidence"]
        sources = data["sources"]
        with st.spinner("Generating reasoning with Gemini…"):
            reasoning = generate_reasoning_with_gemini(article_preview, label, prob, sources)
        st.session_state["last"]["reasoning"] = reasoning
        st.subheader("Reasoning (Gemini)")
        st.write(reasoning)

# ---- Copy reasoning ----
if copy_btn:
    if "last" in st.session_state and st.session_state["last"].get("reasoning"):
        st.code(st.session_state["last"]["reasoning"])
        st.success("Explanation ready — select & copy from the code box.")
    else:
        st.warning("Generate the reasoning first.")
