# streamlit_app.py — NLP Financial Explorer
# NLP Topics: Tokenization, Stopwords, Stemming/Lemmatization, POS Tagging,
#             Sentiment Analysis, NER, Word Frequency, TF-IDF, N-grams,
#             Text Similarity, Text Summarization, Chatbot (Gemini Flash)

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from io import StringIO

# ── NLP ──────────────────────────────────────
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# ── Plotting ─────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go


# ── Google Gemini (new SDK: google-genai) ────
try:
    from google import genai as genai_client
    GENAI_OK = True
except Exception:
    GENAI_OK = False

# ── Finance ───────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_OK = True
except Exception:
    YFINANCE_OK = False

# ── NLTK downloads ───────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ['punkt', 'stopwords', 'averaged_perceptron_tagger',
                'maxent_ne_chunker', 'words', 'wordnet', 'punkt_tab',
                'averaged_perceptron_tagger_eng']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
download_nltk()

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Financial Explorer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif; }
.stApp { background: #f5f0e8; }
.main .block-container { padding-top: 2rem; }

section[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #e8e0d0 !important; }

.nlp-card {
    background: #ffffff;
    border: 2px solid #1a1a2e;
    padding: 20px;
    margin-bottom: 14px;
    box-shadow: 4px 4px 0px #1a1a2e;
}
.nlp-card-accent {
    background: #1a1a2e;
    color: #f5f0e8;
    border: 2px solid #1a1a2e;
    padding: 20px;
    margin-bottom: 14px;
    box-shadow: 4px 4px 0px #e8a020;
    line-height: 1.7;
}
.metric-pill {
    display: inline-block;
    background: #e8a020;
    color: #1a1a2e;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 3px 12px;
    margin: 2px 2px;
    border: 1.5px solid #1a1a2e;
}
.metric-pill.green { background: #4caf7d; color: white; }
.metric-pill.red   { background: #e85020; color: white; }
.metric-pill.blue  { background: #2060e8; color: white; }
.metric-pill.gray  { background: #888;    color: white; }

.token-box {
    display: inline-block;
    background: #f5f0e8;
    border: 1.5px solid #1a1a2e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 9px;
    margin: 2px 2px;
}
.token-box.stop { background: #ffe4e4; border-color: #e85020; color: #e85020; text-decoration: line-through; }
.token-box.stem { background: #e4f4e8; border-color: #4caf7d; }
.token-box.lemma { background: #dce8ff; border-color: #2060e8; }

.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #1a1a2e;
    line-height: 1.1;
    margin-bottom: 4px;
}
.page-badge {
    display: inline-block;
    background: #e8a020;
    color: #1a1a2e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 3px 12px;
    border: 1.5px solid #1a1a2e;
    margin-bottom: 18px;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.sec-divider {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 3px solid #1a1a2e;
    padding-bottom: 6px;
    margin: 24px 0 14px 0;
}
.info-box {
    background: #fffbf0;
    border-left: 4px solid #e8a020;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #1a1a2e;
    line-height: 1.6;
}
.sentiment-bar-wrap {
    background: #e8e0d0;
    border: 1.5px solid #1a1a2e;
    height: 16px;
    width: 100%;
    margin: 6px 0;
}
.chat-user {
    background: #1a1a2e;
    color: #f5f0e8;
    padding: 10px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    max-width: 82%;
    border-radius: 0 12px 12px 12px;
}
.chat-bot {
    background: #ffffff;
    color: #1a1a2e;
    border: 2px solid #1a1a2e;
    padding: 10px 16px;
    margin: 8px 0 8px auto;
    font-size: 0.88rem;
    max-width: 88%;
    border-radius: 12px 0 12px 12px;
    box-shadow: 3px 3px 0px #e8a020;
}
.ner-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.73rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    margin: 2px;
}
textarea, input[type="text"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GEMINI AI CLIENT (Google — Free Forever)
# ─────────────────────────────────────────────
def get_gemini_key():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except (KeyError, FileNotFoundError, Exception):
            pass
    return key.strip() if key else ""

def ai_chat(messages, system="You are a helpful financial analyst.", max_tokens=700):
    api_key = get_gemini_key()
    if not api_key:
        return "⚠️ Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets. Get a FREE key at: aistudio.google.com/app/apikey"
    if not GENAI_OK:
        return "⚠️ google-genai package not installed. Check requirements.txt"
    try:
        client = genai_client.Client(api_key=api_key)
        # Build prompt: system + conversation history + last user message
        full_prompt = system + "\n\n"
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role_label}: {msg['content']}\n"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"

# legacy alias so rest of code works unchanged
groq_chat = ai_chat
groq_client = bool(get_gemini_key())

# ─────────────────────────────────────────────
# SAMPLE FINANCIAL TEXTS
# ─────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Apple Earnings (Positive)": """Apple Inc. reported record-breaking quarterly earnings today, surpassing analyst expectations by a significant margin. The company posted revenue of $119.6 billion, up 9% year-over-year, driven by strong iPhone 15 sales and growth in Services. CEO Tim Cook expressed confidence in Apple's innovation pipeline, highlighting upcoming AI features in iOS. The stock surged 4.2% in after-hours trading following the announcement. Investors are optimistic about the company's expansion into augmented reality and financial services.""",

    "Bank Crisis (Negative)": """Silicon Valley Bank collapsed today in the largest U.S. bank failure since the 2008 financial crisis. Federal regulators seized the bank after a catastrophic run on deposits wiped out $42 billion in a single day. The collapse sent shockwaves through the tech startup ecosystem, with thousands of companies unable to access payroll funds. Treasury Secretary Janet Yellen held emergency meetings with banking regulators. Shares of other regional banks plummeted as contagion fears spread across markets.""",

    "Tesla Report (Neutral)": """Tesla delivered 484,507 vehicles in the third quarter, slightly below analyst expectations of 490,000 units. The electric vehicle maker maintained its annual delivery target despite increased competition from Chinese manufacturers including BYD. Production at the Shanghai Gigafactory continues at full capacity. Elon Musk commented on the competitive landscape during an earnings call. Analysts have mixed views on Tesla's near-term growth prospects given macroeconomic headwinds.""",

    "Fed Rate Decision": """The Federal Reserve raised interest rates by 25 basis points today, bringing the federal funds rate to a 22-year high of 5.25%. Chairman Jerome Powell signaled the possibility of additional hikes if inflation remains above the 2% target. Bond yields rose sharply following the announcement, with the 10-year Treasury yield climbing to 4.8%. Markets reacted with the S&P 500 dropping 1.2% and technology stocks bearing the brunt of the selloff. Mortgage rates are expected to remain elevated, dampening the housing market.""",
}

# ─────────────────────────────────────────────
# NLP UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def get_tokens(text):
    try:
        return word_tokenize(text.lower())
    except Exception:
        return text.lower().split()

def get_sentences(text):
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in text.split('.') if len(s.strip()) > 10]

def get_stopwords_set():
    try:
        return set(stopwords.words('english'))
    except Exception:
        return {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were'}

def get_sentiment(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    label = "Positive" if pol > 0.1 else ("Negative" if pol < -0.1 else "Neutral")
    return pol, sub, label

def get_ner_entities(text):
    entities = {"ORG": [], "GPE": [], "PERSON": [], "MONEY": [], "PCT": []}
    entities["MONEY"] = list(set(re.findall(r'\$[\d,\.]+\s*(?:billion|million|trillion)?|\d+(?:\.\d+)?\s*(?:billion|million)\s*dollars?', text, re.I)))
    entities["PCT"]   = list(set(re.findall(r'\d+(?:\.\d+)?%', text)))
    cap_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    known_orgs    = ['Apple','Tesla','Google','Microsoft','Amazon','Meta','Netflix','Nvidia','Bank','Federal Reserve','FDIC','Treasury','Inc','Corp','BYD','Gigafactory','Silicon Valley Bank']
    known_gpe     = ['U.S.','United States','China','Europe','Shanghai','New York','Washington']
    known_persons = ['Powell','Musk','Cook','Yellen','Biden','Jerome','Tim','Elon','Janet']
    for w in cap_words:
        if   any(o.lower() in w.lower() for o in known_orgs)    and w not in entities["ORG"]:    entities["ORG"].append(w)
        elif any(g.lower() in w.lower() for g in known_gpe)     and w not in entities["GPE"]:    entities["GPE"].append(w)
        elif any(p.lower() in w.lower() for p in known_persons) and w not in entities["PERSON"]: entities["PERSON"].append(w)
    for k in entities:
        entities[k] = list(dict.fromkeys(entities[k]))[:8]
    return entities

def get_tfidf_keywords(text, top_n=15):
    sents = get_sentences(text)
    if len(sents) < 2: sents = [text, text]
    try:
        vec = TfidfVectorizer(stop_words='english', max_features=100)
        mat = vec.fit_transform(sents)
        scores = np.array(mat.sum(axis=0)).flatten()
        words = vec.get_feature_names_out()
        idx = np.argsort(-scores)[:top_n]
        return [(words[i], float(scores[i])) for i in idx]
    except Exception:
        return []

def compute_similarity(texts):
    try:
        vec = TfidfVectorizer(stop_words='english')
        mat = vec.fit_transform(texts)
        return cosine_similarity(mat)
    except Exception:
        return None

# ─────────────────────────────────────────────
# TEXT INPUT WIDGET
# ─────────────────────────────────────────────
def text_input_widget(key_prefix, default="Apple Earnings (Positive)"):
    c1, c2 = st.columns([1, 3])
    with c1:
        sample = st.selectbox("📄 Sample text", list(SAMPLE_TEXTS.keys()),
                              index=list(SAMPLE_TEXTS.keys()).index(default),
                              key=f"{key_prefix}_sample")
    with c2:
        text = st.text_area("Or type / paste your financial text here",
                             value=SAMPLE_TEXTS.get(sample, ""),
                             height=120, key=f"{key_prefix}_text",
                             placeholder="Paste any financial news, earnings report, or market commentary...")
    return text.strip()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 22px 0;border-bottom:1px solid #333;margin-bottom:18px;'>
        <div style='font-size:2.2rem;'>📰</div>
        <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:800;color:#e8a020;margin-top:6px;'>NLP Financial<br>Explorer</div>
        <div style='font-size:0.65rem;color:#aaa;margin-top:4px;letter-spacing:2px;text-transform:uppercase;'>All NLP Topics Covered</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "✂️  Text Preprocessing",
        "😊  Sentiment Analysis",
        "🏷️  Named Entity Recognition",
        "📊  Word Frequency & TF-IDF",
        "🔍  Text Similarity",
        "📝  Text Summarization",
        "🤖  Financial Chatbot",
        "📈  Stock Data + NLP",
    ], label_visibility="collapsed")

    st.markdown("---")
    try:
        _has_key = bool(st.secrets.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", ""))
    except Exception:
        _has_key = bool(os.environ.get("GEMINI_API_KEY", ""))
    status = "✅ Connected" if _has_key else "❌ Add key to Secrets"
    st.markdown(f"<div style='font-size:0.73rem;color:#aaa;'>🔑 Gemini API: {status}</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.73rem;color:#888;margin-top:4px;'>📦 NLTK · TextBlob · sklearn · yfinance</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="page-title">NLP Financial Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">Natural Language Processing · Finance · All Topics</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <strong>Natural Language Processing (NLP)</strong> is the branch of AI that enables computers to understand and work with human language.
    In finance, NLP powers sentiment analysis of earnings calls, extraction of key figures from reports, news summarization, and intelligent chatbots.
    This app demonstrates every major NLP concept interactively, using real financial text.
    </div>""", unsafe_allow_html=True)

    topics = [
        ("✂️", "Text Preprocessing",        "Tokenization · Stopwords · Stemming · Lemmatization · POS",   "The foundation — clean and structure raw text before any analysis."),
        ("😊", "Sentiment Analysis",         "Polarity · Subjectivity · TextBlob · Per-sentence scoring",   "Detect positive/negative tone in financial news and reports."),
        ("🏷️", "Named Entity Recognition",   "ORG · GPE · PERSON · MONEY · Highlighted text",              "Auto-extract companies, people, locations, and dollar amounts."),
        ("📊", "Word Frequency & TF-IDF",    "Bag of Words · Term Frequency · TF-IDF · N-grams",            "Find the most important words and phrases statistically."),
        ("🔍", "Text Similarity",            "Cosine Similarity · TF-IDF Vectors · Heatmap",                "Measure how similar two financial documents are."),
        ("📝", "Text Summarization",         "Extractive (TF-IDF) · Abstractive (Gemini Flash)",            "Condense long financial reports into key bullet points."),
        ("🤖", "Financial Chatbot",          "LLM · Gemini Flash · Context-aware Q&A",                     "Ask any financial question in plain English."),
        ("📈", "Stock Data + NLP",           "yfinance · Real prices · Sentiment on news",                  "Combine live stock data with NLP analysis."),
    ]

    for i in range(0, len(topics), 2):
        c1, c2 = st.columns(2)
        for col, j in zip([c1, c2], [i, i+1]):
            if j < len(topics):
                icon, name, tags, desc = topics[j]
                with col:
                    st.markdown(f"""<div class="nlp-card" style="min-height:120px;">
                    <div style="font-size:1.5rem;margin-bottom:6px;">{icon}</div>
                    <div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#1a1a2e;margin-bottom:5px;">{name}</div>
                    <div style="font-size:0.72rem;color:#888;margin-bottom:8px;font-family:IBM Plex Mono,monospace;">{tags}</div>
                    <div style="font-size:0.82rem;color:#444;line-height:1.5;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-divider">The NLP Pipeline</div>', unsafe_allow_html=True)
    steps = [("1", "Raw Text", "#e8a020"), ("2", "Tokenize", "#ffffff"), ("3", "Clean", "#ffffff"),
             ("4", "Vectorize", "#ffffff"), ("5", "Model", "#ffffff"), ("6", "Output", "#1a1a2e")]
    cols = st.columns(len(steps))
    for col, (num, step, bg) in zip(cols, steps):
        tc = "#1a1a2e" if bg != "#1a1a2e" else "#f5f0e8"
        with col:
            st.markdown(f"""<div style="background:{bg};color:{tc};border:2px solid #1a1a2e;padding:12px 6px;
            text-align:center;font-family:IBM Plex Mono,monospace;font-size:0.72rem;font-weight:600;
            box-shadow:3px 3px 0 #1a1a2e;"><div style="font-size:1.1rem;">{num}</div>{step}</div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="sec-divider">Try It Now — Live Sentiment Check</div>', unsafe_allow_html=True)
    quick_text = st.text_input("Type any financial sentence and see instant sentiment:",
                                placeholder="e.g. Apple smashed earnings expectations today")
    if quick_text:
        pol, sub, lbl = get_sentiment(quick_text)
        color = "#4caf7d" if lbl=="Positive" else ("#e85020" if lbl=="Negative" else "#888")
        st.markdown(f"""<div class="nlp-card" style="border-color:{color};box-shadow:4px 4px 0 {color};">
        <span style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:{color};">{lbl}</span>
        &nbsp;&nbsp;<span style="font-family:IBM Plex Mono,monospace;font-size:0.9rem;color:#666;">polarity: {pol:+.3f} | subjectivity: {sub:.3f}</span>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════
elif page == "✂️  Text Preprocessing":
    st.markdown('<div class="page-title">Text Preprocessing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">NLP Foundation · Step-by-Step Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">Before any NLP algorithm runs, raw text must be <strong>cleaned and normalized</strong>.
    This page walks through every preprocessing step — tokenization, stopword removal, stemming, lemmatization, and POS tagging —
    so you can see exactly what happens to text before analysis.</div>""", unsafe_allow_html=True)

    text = text_input_widget("prep")
    if not text:
        st.warning("Please enter some text.")
        st.stop()

    tokens    = get_tokens(text)
    sw        = get_stopwords_set()
    alpha_tok = [t for t in tokens if t.isalpha()]
    filtered  = [t for t in alpha_tok if t not in sw]
    stemmed   = [PorterStemmer().stem(t) for t in filtered]
    lemmatized= [WordNetLemmatizer().lemmatize(t) for t in filtered]
    sentences = get_sentences(text)

    # Stats row
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in zip([c1,c2,c3,c4,c5],
        [len(sentences), len(tokens), len(filtered), len(set(filtered)), round(len(filtered)/max(len(sentences),1),1)],
        ["Sentences","Total Tokens","After Stopwords","Unique Words","Avg/Sentence"]):
        with col:
            st.markdown(f"""<div class="nlp-card" style="text-align:center;padding:14px;">
            <div style="font-family:IBM Plex Mono,monospace;font-size:1.7rem;font-weight:700;color:#e8a020;">{val}</div>
            <div style="font-size:0.7rem;color:#666;text-transform:uppercase;letter-spacing:1px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # STEP 1: Tokenization
    st.markdown('<div class="sec-divider">Step 1 — Tokenization</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>Tokenization</strong> splits text into individual words and punctuation marks.
    Each item is called a <em>token</em>. This is always the very first step.</div>""", unsafe_allow_html=True)
    tok_html = " ".join([f'<span class="token-box">{t}</span>' for t in tokens[:60]])
    if len(tokens) > 60: tok_html += f'<span style="color:#888;font-size:0.8rem;"> ... +{len(tokens)-60} more</span>'
    st.markdown(tok_html, unsafe_allow_html=True)

    # STEP 2: Stopword Removal
    st.markdown('<div class="sec-divider">Step 2 — Stopword Removal</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>Stopwords</strong> are common words (the, is, at, for...) that carry little meaning.
    <span style="color:#e85020;font-weight:600;">Red strikethrough = removed</span>.
    Removing them reduces noise.</div>""", unsafe_allow_html=True)
    stop_html = "".join([
        f'<span class="token-box stop">{t}</span>' if t in sw else f'<span class="token-box">{t}</span>'
        for t in alpha_tok[:80]
    ])
    st.markdown(stop_html, unsafe_allow_html=True)
    removed_pct = round((1 - len(filtered)/max(len(alpha_tok),1))*100)
    st.markdown(f'<span class="metric-pill red">{removed_pct}% of words removed as stopwords</span>', unsafe_allow_html=True)

    # STEP 3: Stemming vs Lemmatization
    st.markdown('<div class="sec-divider">Step 3 — Stemming vs Lemmatization</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="nlp-card">
        <div style="font-family:Syne,sans-serif;font-weight:700;margin-bottom:8px;">⚡ Stemming (Porter)</div>
        <div style="font-size:0.8rem;color:#555;margin-bottom:10px;">Fast but crude — chops word endings. "running"→"run", "studies"→"studi"</div>"""
        + " ".join([f'<span class="token-box stem">{s}</span>' for s in stemmed[:35]])
        + "</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="nlp-card">
        <div style="font-family:Syne,sans-serif;font-weight:700;margin-bottom:8px;">✨ Lemmatization (WordNet)</div>
        <div style="font-size:0.8rem;color:#555;margin-bottom:10px;">Slower but correct dictionary form. "better"→"good", "studies"→"study"</div>"""
        + " ".join([f'<span class="token-box lemma">{l}</span>' for l in lemmatized[:35]])
        + "</div>", unsafe_allow_html=True)

    if filtered:
        st.markdown('<div class="sec-divider">Comparison Table (first 20 words)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Original": filtered[:20], "Stemmed": stemmed[:20], "Lemmatized": lemmatized[:20]}),
                     use_container_width=True, hide_index=True)

    # STEP 4: POS Tagging
    st.markdown('<div class="sec-divider">Step 4 — Part-of-Speech (POS) Tagging</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>POS tagging</strong> labels each word with its grammatical role:
    Noun (NN), Verb (VB), Adjective (JJ), Adverb (RB), Proper Noun (NNP). Essential for NER and parsing.</div>""", unsafe_allow_html=True)
    try:
        pos_tokens = word_tokenize(text[:500])
        pos_tagged = pos_tag(pos_tokens)
        pos_colors = {"NN": "#e8a020", "VB": "#2060e8", "JJ": "#4caf7d", "RB": "#9c27b0", "NNP": "#e85020"}
        pos_html = ""
        for word, tag in pos_tagged[:50]:
            if not word.isalpha(): continue
            c = pos_colors.get(tag[:2], "#999")
            pos_html += f'<span style="display:inline-block;margin:2px;padding:2px 7px;background:#f5f0e8;border:1.5px solid {c};font-family:IBM Plex Mono,monospace;font-size:0.7rem;"><span style="color:{c};font-weight:700;">{tag}</span> {word}</span>'
        st.markdown(pos_html, unsafe_allow_html=True)

        pos_counts = Counter([tag[:2] for _, tag in pos_tagged if tag[:2] in pos_colors])
        if pos_counts:
            fig_pos = px.bar(x=list(pos_colors.keys()),
                            y=[pos_counts.get(k,0) for k in pos_colors.keys()],
                            title="POS Tag Distribution",
                            labels={"x":"POS","y":"Count"},
                            color=list(pos_colors.keys()),
                            color_discrete_sequence=list(pos_colors.values()))
            fig_pos.update_layout(showlegend=False, height=260,
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(245,240,232,1)',
                                  font=dict(family="IBM Plex Mono"))
            st.plotly_chart(fig_pos, use_container_width=True)
    except Exception as e:
        st.info(f"POS tagging: {e}")

# ═══════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "😊  Sentiment Analysis":
    st.markdown('<div class="page-title">Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">Opinion Mining · Polarity · Subjectivity</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box"><strong>Sentiment Analysis</strong> automatically detects the emotional tone of text —
    positive, negative, or neutral. In finance, it's used to gauge market sentiment from news articles,
    earnings calls, and analyst reports to predict investor behavior and price movements.</div>""", unsafe_allow_html=True)

    text = text_input_widget("sent")
    if not text: st.warning("Please enter text."); st.stop()

    polarity, subjectivity, label = get_sentiment(text)
    s_color = "#4caf7d" if label=="Positive" else ("#e85020" if label=="Negative" else "#888")
    s_icon  = "😊" if label=="Positive" else ("😟" if label=="Negative" else "😐")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="nlp-card" style="text-align:center;border-color:{s_color};box-shadow:4px 4px 0 {s_color};">
        <div style="font-size:2.8rem;">{s_icon}</div>
        <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:{s_color};">{label}</div>
        <div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:1px;">Overall Sentiment</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        pol_pct = int((polarity+1)/2*100)
        fill = "#4caf7d" if polarity>0.1 else ("#e85020" if polarity<-0.1 else "#888")
        st.markdown(f"""<div class="nlp-card">
        <div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:700;">{polarity:+.3f}</div>
        <div style="font-size:0.7rem;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Polarity Score (-1 to +1)</div>
        <div class="sentiment-bar-wrap"><div style="width:{pol_pct}%;height:100%;background:{fill};"></div></div>
        <div style="font-size:0.7rem;color:#888;">-1 very negative → +1 very positive</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        sub_pct = int(subjectivity*100)
        st.markdown(f"""<div class="nlp-card">
        <div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;font-weight:700;">{subjectivity:.3f}</div>
        <div style="font-size:0.7rem;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Subjectivity Score (0 to 1)</div>
        <div class="sentiment-bar-wrap"><div style="width:{sub_pct}%;height:100%;background:#2060e8;"></div></div>
        <div style="font-size:0.7rem;color:#888;">0 = objective fact → 1 = personal opinion</div>
        </div>""", unsafe_allow_html=True)

    # Per-sentence
    st.markdown('<div class="sec-divider">Sentence-by-Sentence Breakdown</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">Each sentence is scored individually — revealing which parts of a financial 
    document are driving positive or negative tone.</div>""", unsafe_allow_html=True)

    sents = get_sentences(text)
    sent_data = []
    for s in sents:
        if len(s.split()) < 4: continue
        p, sb, l = get_sentiment(s)
        sent_data.append({"Sentence": s, "Polarity": round(p,3), "Label": l})

    for row in sent_data:
        border = "#4caf7d" if row["Label"]=="Positive" else ("#e85020" if row["Label"]=="Negative" else "#888")
        icon   = "📈" if row["Label"]=="Positive" else ("📉" if row["Label"]=="Negative" else "➡️")
        st.markdown(f"""<div style="border-left:4px solid {border};background:white;padding:10px 16px;
        margin:5px 0;font-size:0.86rem;color:#1a1a2e;">
        {icon} {row["Sentence"]}
        <span style="float:right;font-family:IBM Plex Mono,monospace;font-size:0.76rem;color:{border};font-weight:600;">
        {row["Polarity"]:+.3f}</span></div>""", unsafe_allow_html=True)

    if sent_data:
        sdf = pd.DataFrame(sent_data)
        c1, c2 = st.columns(2)
        with c1:
            lc = Counter([r["Label"] for r in sent_data])
            fig_p = px.pie(values=list(lc.values()), names=list(lc.keys()), title="Sentence Distribution",
                          color=list(lc.keys()),
                          color_discrete_map={"Positive":"#4caf7d","Negative":"#e85020","Neutral":"#888"})
            fig_p.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', font=dict(family="IBM Plex Mono"))
            st.plotly_chart(fig_p, use_container_width=True)
        with c2:
            fig_b = px.bar(sdf, y="Polarity", color="Label",
                          color_discrete_map={"Positive":"#4caf7d","Negative":"#e85020","Neutral":"#888"},
                          title="Polarity Per Sentence")
            fig_b.add_hline(y=0, line_color="#1a1a2e", line_width=1.5)
            fig_b.update_layout(height=280, showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(245,240,232,1)', font=dict(family="IBM Plex Mono"))
            st.plotly_chart(fig_b, use_container_width=True)

    st.markdown('<div class="sec-divider">Compare All Sample Texts</div>', unsafe_allow_html=True)
    cmp = [{"Text":n, "Polarity":round(get_sentiment(t)[0],3), "Sentiment":get_sentiment(t)[2]} for n,t in SAMPLE_TEXTS.items()]
    cdf = pd.DataFrame(cmp)
    fig_cmp = px.bar(cdf, x="Text", y="Polarity", color="Sentiment",
                    color_discrete_map={"Positive":"#4caf7d","Negative":"#e85020","Neutral":"#888"},
                    title="Sentiment Comparison Across All Samples")
    fig_cmp.add_hline(y=0, line_color="#1a1a2e", line_width=1.5)
    fig_cmp.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(245,240,232,1)', font=dict(family="IBM Plex Mono"))
    st.plotly_chart(fig_cmp, use_container_width=True)

# ═══════════════════════════════════════════════════════
# NER
# ═══════════════════════════════════════════════════════
elif page == "🏷️  Named Entity Recognition":
    st.markdown('<div class="page-title">Named Entity Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">NER · Information Extraction</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box"><strong>NER</strong> identifies and classifies real-world entities in text:
    <strong>companies (ORG)</strong>, <strong>locations (GPE)</strong>, <strong>people (PERSON)</strong>,
    <strong>money (MONEY)</strong>, and <strong>percentages</strong>.
    In finance, NER extracts structured data from unstructured documents automatically — 
    saving analysts hours of manual work.</div>""", unsafe_allow_html=True)

    text = text_input_widget("ner")
    if not text: st.warning("Please enter text."); st.stop()

    entities = get_ner_entities(text)
    tag_styles = {
        "ORG":    ("🏢", "#fff3cd", "#e8a020", "Organization"),
        "GPE":    ("🌍", "#cce5ff", "#2060e8", "Location/Country"),
        "PERSON": ("👤", "#f8d7da", "#e85020", "Person"),
        "MONEY":  ("💰", "#d4edda", "#4caf7d", "Money Amount"),
        "PCT":    ("📌", "#e8e0d0", "#666666", "Percentage"),
    }

    has_ents = any(entities[k] for k in entities)
    if not has_ents:
        st.info("No named entities matched. Try the 'Apple Earnings' or 'Bank Crisis' sample.")

    for etype, (icon, bg, border, label) in tag_styles.items():
        ents = entities.get(etype, [])
        if not ents: continue
        tags_html = "".join([f'<span class="ner-tag" style="background:{bg};color:{border};border:1.5px solid {border};">{e}</span>' for e in ents])
        st.markdown(f"""<div class="nlp-card" style="padding:14px;margin-bottom:8px;">
        <div style="font-size:0.73rem;font-family:IBM Plex Mono,monospace;font-weight:700;color:{border};
        text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">{icon} {label} ({etype})</div>
        {tags_html}</div>""", unsafe_allow_html=True)

    if has_ents:
        # Highlighted text
        st.markdown('<div class="sec-divider">Highlighted Text View</div>', unsafe_allow_html=True)
        highlighted = text
        for etype, ents_list in entities.items():
            _, bg, border, _ = tag_styles[etype]
            for ent in ents_list:
                highlighted = highlighted.replace(ent,
                    f'<mark style="background:{bg};border:1px solid {border};padding:1px 4px;font-weight:600;">'
                    f'{ent}<sup style="font-size:0.58rem;color:{border};"> {etype}</sup></mark>')
        st.markdown(f'<div style="background:white;border:2px solid #1a1a2e;padding:16px;font-size:0.88rem;line-height:1.9;">{highlighted}</div>',
                    unsafe_allow_html=True)

        # Chart
        ent_counts = {tag_styles[k][3]: len(v) for k, v in entities.items() if v}
        if ent_counts:
            fig_ner = px.bar(x=list(ent_counts.keys()), y=list(ent_counts.values()),
                            title="Entity Type Counts",
                            color=list(ent_counts.keys()),
                            color_discrete_sequence=["#e8a020","#2060e8","#e85020","#4caf7d","#888"])
            fig_ner.update_layout(showlegend=False, height=240, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(245,240,232,1)', font=dict(family="IBM Plex Mono"))
            st.plotly_chart(fig_ner, use_container_width=True)

# ═══════════════════════════════════════════════════════
# WORD FREQUENCY & TF-IDF
# ═══════════════════════════════════════════════════════
elif page == "📊  Word Frequency & TF-IDF":
    st.markdown('<div class="page-title">Word Frequency & TF-IDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">Bag of Words · TF-IDF · N-grams · Keywords</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <strong>Word Frequency</strong> counts raw occurrences.
    <strong>TF-IDF</strong> (Term Frequency–Inverse Document Frequency) scores words by importance within
    a document relative to all documents — making it excellent for <em>keyword extraction</em>.
    <strong>N-grams</strong> capture meaningful phrases like "earnings per share" or "interest rate hike".
    </div>""", unsafe_allow_html=True)

    text = text_input_widget("freq")
    if not text: st.warning("Please enter text."); st.stop()

    tokens   = get_tokens(text)
    sw       = get_stopwords_set()
    filtered = [t for t in tokens if t.isalpha() and t not in sw and len(t)>2]
    freq     = Counter(filtered)
    top20    = freq.most_common(20)
    tfidf_kw = get_tfidf_keywords(text, 20)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-divider">Top Word Frequencies</div>', unsafe_allow_html=True)
        if top20:
            df_f = pd.DataFrame(top20, columns=["Word","Count"])
            fig_f = px.bar(df_f, x="Count", y="Word", orientation='h', title="Most Frequent Words",
                          color="Count", color_continuous_scale=["#e8e0d0","#e8a020","#1a1a2e"])
            fig_f.update_layout(height=480, yaxis={'categoryorder':'total ascending'},
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(245,240,232,1)',
                                showlegend=False, coloraxis_showscale=False, font=dict(family="IBM Plex Mono"))
            st.plotly_chart(fig_f, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-divider">TF-IDF Keywords</div>', unsafe_allow_html=True)
        if tfidf_kw:
            df_t = pd.DataFrame(tfidf_kw, columns=["Keyword","TF-IDF"])
            fig_t = px.bar(df_t, x="TF-IDF", y="Keyword", orientation='h', title="Keywords by TF-IDF Score",
                          color="TF-IDF", color_continuous_scale=["#e8e0d0","#2060e8","#1a1a2e"])
            fig_t.update_layout(height=480, yaxis={'categoryorder':'total ascending'},
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(245,240,232,1)',
                                showlegend=False, coloraxis_showscale=False, font=dict(family="IBM Plex Mono"))
            st.plotly_chart(fig_t, use_container_width=True)

    # Word cloud
    st.markdown('<div class="sec-divider">Word Cloud</div>', unsafe_allow_html=True)
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(width=900, height=320, background_color="#f5f0e8",
                       colormap="YlOrBr", max_words=70).generate(" ".join(filtered))
        fig_wc, ax = plt.subplots(figsize=(11, 3.5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        fig_wc.patch.set_facecolor('#f5f0e8')
        st.pyplot(fig_wc)
    except ImportError:
        max_c = top20[0][1] if top20 else 1
        cloud = "".join([f'<span style="font-size:{0.8+(c/max_c)*1.6:.1f}rem;opacity:{0.4+(c/max_c)*0.6:.2f};color:#1a1a2e;margin:3px 5px;display:inline-block;font-family:Syne,sans-serif;font-weight:700;">{w}</span>' for w,c in freq.most_common(40)])
        st.markdown(f'<div style="background:white;border:2px solid #1a1a2e;padding:20px;">{cloud}</div>', unsafe_allow_html=True)

    # N-grams
    st.markdown('<div class="sec-divider">N-grams (Bigrams & Trigrams)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box"><strong>N-grams</strong> are sequences of N consecutive words. 
    They capture phrases and context that single words miss — e.g., "interest rate", "earnings per share".</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, n, title, color in [(c1, (2,2), "Top Bigrams", "#e8a020"), (c2, (3,3), "Top Trigrams", "#2060e8")]:
        with col:
            try:
                v = CountVectorizer(ngram_range=n, stop_words='english', max_features=10)
                v.fit_transform([text])
                ngram_counts = dict(zip(v.get_feature_names_out(), v.transform([text]).toarray()[0]))
                df_ng = pd.DataFrame(sorted(ngram_counts.items(), key=lambda x:-x[1])[:10], columns=["Phrase","Count"])
                fig_ng = px.bar(df_ng, x="Count", y="Phrase", orientation='h', title=title,
                               color_discrete_sequence=[color])
                fig_ng.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(245,240,232,1)', font=dict(family="IBM Plex Mono"))
                st.plotly_chart(fig_ng, use_container_width=True)
            except Exception:
                st.info("Add more text to see n-grams.")

# ═══════════════════════════════════════════════════════
# TEXT SIMILARITY
# ═══════════════════════════════════════════════════════
elif page == "🔍  Text Similarity":
    st.markdown('<div class="page-title">Text Similarity</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">Cosine Similarity · TF-IDF Vectors · Jaccard</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box"><strong>Text Similarity</strong> measures how alike two documents are.
    <strong>Cosine similarity</strong> compares TF-IDF vectors — a score of 0 means completely different,
    1 means identical. Finance uses this for: detecting duplicate filings, clustering related news,
    and finding similar analyst reports.</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Text A**")
        sa = st.selectbox("Sample A", list(SAMPLE_TEXTS.keys()), index=0, key="sia")
        ta = st.text_area("", value=SAMPLE_TEXTS[sa], height=140, key="ta")
    with c2:
        st.markdown("**Text B**")
        sb = st.selectbox("Sample B", list(SAMPLE_TEXTS.keys()), index=1, key="sib")
        tb = st.text_area("", value=SAMPLE_TEXTS[sb], height=140, key="tb")

    if ta and tb:
        sm = compute_similarity([ta, tb])
        if sm is not None:
            score = sm[0][1]
            pct   = int(score*100)
            color = "#4caf7d" if pct>50 else ("#e8a020" if pct>20 else "#e85020")
            st.markdown(f"""<div class="nlp-card" style="text-align:center;padding:28px;border-color:{color};box-shadow:4px 4px 0 {color};">
            <div style="font-family:Syne,sans-serif;font-size:4rem;font-weight:800;color:{color};">{pct}%</div>
            <div style="font-family:IBM Plex Mono,monospace;color:#666;margin-top:4px;">Cosine Similarity</div>
            <div style="margin-top:10px;font-size:0.84rem;color:#444;">
            {"🟢 High — both texts discuss similar financial topics" if pct>50 else ("🟡 Moderate overlap" if pct>20 else "🔴 Low — very different topics")}
            </div></div>""", unsafe_allow_html=True)

            sw = get_stopwords_set()
            wa = {t for t in get_tokens(ta) if t.isalpha() and t not in sw}
            wb = {t for t in get_tokens(tb) if t.isalpha() and t not in sw}
            common = wa & wb
            jaccard = len(common)/len(wa|wb) if (wa|wb) else 0

            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="nlp-card" style="min-height:100px;">
                <div style="font-weight:700;color:#4caf7d;margin-bottom:6px;">✅ Common ({len(common)})</div>
                {"".join([f'<span class="metric-pill green">{w}</span>' for w in list(common)[:14]])}
                </div>""", unsafe_allow_html=True)
            with col2:
                only_a = wa-wb
                st.markdown(f"""<div class="nlp-card" style="min-height:100px;">
                <div style="font-weight:700;color:#2060e8;margin-bottom:6px;">🔵 Only A ({len(only_a)})</div>
                {"".join([f'<span class="metric-pill blue">{w}</span>' for w in list(only_a)[:12]])}
                </div>""", unsafe_allow_html=True)
            with col3:
                only_b = wb-wa
                st.markdown(f"""<div class="nlp-card" style="min-height:100px;">
                <div style="font-weight:700;color:#e85020;margin-bottom:6px;">🔴 Only B ({len(only_b)})</div>
                {"".join([f'<span class="metric-pill red">{w}</span>' for w in list(only_b)[:12]])}
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<span class="metric-pill">Jaccard Similarity: {jaccard:.3f}</span>', unsafe_allow_html=True)

    st.markdown('<div class="sec-divider">All Samples — Similarity Heatmap</div>', unsafe_allow_html=True)
    names  = list(SAMPLE_TEXTS.keys())
    txts   = list(SAMPLE_TEXTS.values())
    sm_all = compute_similarity(txts)
    if sm_all is not None:
        fig_hm = px.imshow(sm_all, x=names, y=names,
                           color_continuous_scale=["#f5f0e8","#e8a020","#1a1a2e"],
                           title="Document Similarity Matrix", zmin=0, zmax=1, text_auto=".2f")
        fig_hm.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', font=dict(family="IBM Plex Mono"))
        st.plotly_chart(fig_hm, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TEXT SUMMARIZATION
# ═══════════════════════════════════════════════════════
elif page == "📝  Text Summarization":
    st.markdown('<div class="page-title">Text Summarization</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">Extractive · Abstractive · AI-Powered</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    <strong>Extractive summarization</strong> selects the most important existing sentences using TF-IDF scores.
    <strong>Abstractive summarization</strong> uses an AI language model to generate entirely new, 
    concise sentences. Finance use: quickly digest earnings reports, SEC filings, and breaking news.
    </div>""", unsafe_allow_html=True)

    text = text_input_widget("summ", "Bank Crisis (Negative)")
    if not text: st.warning("Please enter text."); st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-divider">📌 Extractive (TF-IDF)</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">Picks the top-scoring sentences by word importance.
        <strong>No AI required</strong> — pure statistics.</div>""", unsafe_allow_html=True)
        n_s = st.slider("Sentences to extract", 1, 5, 2, key="es_n")

        sents = get_sentences(text)
        if len(sents) <= n_s:
            st.markdown(f'<div class="nlp-card-accent">{text}</div>', unsafe_allow_html=True)
        else:
            try:
                tv = TfidfVectorizer(stop_words='english')
                tm = tv.fit_transform(sents)
                scores = np.array(tm.sum(axis=1)).flatten()
                idxs = sorted(np.argsort(-scores)[:n_s])
                summary = " ".join([sents[i] for i in idxs])
                compression = round((1-len(summary.split())/max(len(text.split()),1))*100)
                st.markdown(f'<div class="nlp-card-accent">{summary}</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="metric-pill green">{compression}% shorter than original</span>', unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

    with c2:
        st.markdown('<div class="sec-divider">✨ Abstractive (Gemini Flash)</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">AI generates <strong>new sentences</strong> capturing the core meaning.
        More natural and readable than extractive.</div>""", unsafe_allow_html=True)

        style = st.selectbox("Style", [
            "Bullet points (investor view)",
            "One concise paragraph",
            "ELI5 — explain simply",
            "Risk & downside focus"
        ], key="summ_style")

        if st.button("✨ Generate AI Summary", type="primary"):
            prompts = {
                "Bullet points (investor view)": "Summarize in 4-5 bullet points for an investor. Include numbers and key outcomes.",
                "One concise paragraph": "Summarize in one clear, concise paragraph.",
                "ELI5 — explain simply": "Explain this financial text in simple language for a 12-year-old.",
                "Risk & downside focus": "Summarize focusing on risks, threats, and negative implications for investors."
            }
            with st.spinner("Generating..."):
                result = groq_chat(
                    [{"role":"user","content":f"{prompts[style]}\n\nText:\n{text}"}],
                    system="You are a concise financial analyst. Be direct and factual.")
            st.markdown(f'<div class="nlp-card-accent">{result.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Click to generate an AI summary using Gemini Flash.")

# ═══════════════════════════════════════════════════════
# CHATBOT
# ═══════════════════════════════════════════════════════
elif page == "🤖  Financial Chatbot":
    st.markdown('<div class="page-title">Financial Q&A Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">Gemini Flash · Context-Aware · Financial NLP</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">Ask any financial question in plain English.
    Powered by <strong>Google Gemini</strong> — a fast, free AI model with deep financial knowledge.
    Optionally add a context document (e.g. a news article) for document-specific Q&A.
    </div>""", unsafe_allow_html=True)

    if not get_gemini_key():
        st.markdown("""<div class="nlp-card" style="border-color:#e85020;">
        <div style="color:#e85020;font-weight:700;font-size:1.1rem;margin-bottom:8px;">⚠️ API Key Required</div>
        <div style="font-size:0.86rem;color:#444;">Go to your Streamlit app → <strong>Settings → Secrets</strong><br>
        Add: <code style="background:#f5f0e8;padding:2px 6px;">GEMINI_API_KEY = "your_key_here"</code><br>
        Get a FREE key at <strong>aistudio.google.com/app/apikey</strong>
        </div></div>""", unsafe_allow_html=True)
        st.stop()

    with st.expander("📄 Provide a context document (for document Q&A)"):
        ctx_choice = st.selectbox("Use sample text", ["None"]+list(SAMPLE_TEXTS.keys()))
        ctx_text = SAMPLE_TEXTS.get(ctx_choice,"") if ctx_choice!="None" else ""
        ctx_custom = st.text_area("Or paste your own context", height=90, key="ctx_paste")
        if ctx_custom: ctx_text = ctx_custom

    # Quick question buttons
    st.markdown('<div class="sec-divider">Quick Questions</div>', unsafe_allow_html=True)
    quick = ["What is P/E ratio?","What does a Fed rate hike mean for stocks?",
             "Explain earnings per share","What is inflation?",
             "What is a short squeeze?","How do bonds work?"]
    cols = st.columns(3)
    for i, q in enumerate(quick):
        with cols[i%3]:
            if st.button(q, key=f"qb_{i}", use_container_width=True):
                st.session_state["_pending_q"] = q

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        st.markdown('<div class="sec-divider">Conversation</div>', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {msg["content"].replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-divider">Your Question</div>', unsafe_allow_html=True)
    default_q = st.session_state.pop("_pending_q", "")
    user_q = st.text_input("Ask anything financial:", value=default_q,
                           placeholder="e.g. What caused the 2008 financial crisis?", key="chat_q")

    col1, col2 = st.columns([1,6])
    with col1:
        send = st.button("Send →", type="primary")
    with col2:
        if st.button("🗑 Clear"):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_q:
        sys_prompt = "You are an expert financial analyst and educator. Answer clearly with line breaks. Use simple language when possible."
        if ctx_text:
            sys_prompt += f"\n\nUser-provided context:\n{ctx_text[:2000]}"
        msgs = st.session_state.chat_history[-6:] + [{"role":"user","content":user_q}]
        with st.spinner("Thinking..."):
            reply = groq_chat(msgs, system=sys_prompt, max_tokens=600)
        st.session_state.chat_history.append({"role":"user","content":user_q})
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

# ═══════════════════════════════════════════════════════
# STOCK DATA + NLP
# ═══════════════════════════════════════════════════════
elif page == "📈  Stock Data + NLP":
    st.markdown('<div class="page-title">Stock Data + NLP</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-badge">yfinance · Live Prices · Sentiment Analysis</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">Combine <strong>real stock market data</strong> with NLP analysis.
    See the stock price chart, then analyze any news article about that company using sentiment analysis,
    keyword extraction, and NER — all in one place.</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="TSLA, MSFT, GOOGL").upper()
        period = st.selectbox("Period", ["1mo","3mo","6mo","1y"], index=1)

    if not YFINANCE_OK:
        st.warning("yfinance not installed. Add `yfinance` to requirements.txt.")
        st.stop()

    if ticker:
        try:
            with st.spinner(f"Fetching {ticker}..."):
                stock = yf.Ticker(ticker)
                hist  = stock.history(period=period)
                info  = stock.info

            if hist.empty:
                st.error(f"No data for {ticker}. Check the ticker symbol.")
            else:
                price   = hist['Close'].iloc[-1]
                change  = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                chg_pct = change/hist['Close'].iloc[-2]*100

                with c1:
                    color = "#4caf7d" if change>=0 else "#e85020"
                    st.markdown(f"""<div class="nlp-card" style="text-align:center;border-color:{color};box-shadow:4px 4px 0 {color};">
                    <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:1px;">{ticker}</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:2rem;font-weight:700;color:#1a1a2e;">${price:.2f}</div>
                    <div style="font-family:IBM Plex Mono,monospace;color:{color};font-size:1rem;">{change:+.2f} ({chg_pct:+.2f}%)</div>
                    </div>""", unsafe_allow_html=True)

                    mcap = info.get("marketCap")
                    if mcap:
                        mcap_s = f"${mcap/1e12:.2f}T" if mcap>1e12 else f"${mcap/1e9:.1f}B"
                        st.markdown(f'<span class="metric-pill">{mcap_s} Market Cap</span>', unsafe_allow_html=True)
                    sector = info.get("sector")
                    if sector:
                        st.markdown(f'<span class="metric-pill blue">{sector}</span>', unsafe_allow_html=True)

                with c2:
                    fig_p = go.Figure(go.Candlestick(
                        x=hist.index, open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'], name=ticker,
                        increasing_line_color='#4caf7d', decreasing_line_color='#e85020'))
                    fig_p.update_layout(title=f"{ticker} — {period}", xaxis_rangeslider_visible=False,
                                        height=340, paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(245,240,232,1)', font=dict(family="IBM Plex Mono"))
                    st.plotly_chart(fig_p, use_container_width=True)

                # Company description NLP
                desc = info.get("longBusinessSummary","")
                if desc:
                    st.markdown('<div class="sec-divider">NLP on Company Description</div>', unsafe_allow_html=True)
                    pol, sub, lbl = get_sentiment(desc)
                    kws = [k for k,_ in get_tfidf_keywords(desc, 8)]
                    ents = get_ner_entities(desc)
                    sc = "#4caf7d" if lbl=="Positive" else ("#e85020" if lbl=="Negative" else "#888")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""<div class="nlp-card" style="text-align:center;">
                        <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:{sc};">{lbl}</div>
                        <div style="font-family:IBM Plex Mono,monospace;font-size:0.85rem;">Polarity: {pol:+.3f}</div>
                        <div style="font-size:0.7rem;color:#888;">Description Sentiment</div></div>""", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""<div class="nlp-card">
                        <div style="font-weight:700;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">🔑 Keywords</div>
                        {"".join([f'<span class="metric-pill">{k}</span>' for k in kws])}
                        </div>""", unsafe_allow_html=True)
                    with col3:
                        orgs = ents.get("ORG",[])[:5]
                        st.markdown(f"""<div class="nlp-card">
                        <div style="font-weight:700;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">🏢 Organizations</div>
                        {"".join([f'<span class="metric-pill">{o}</span>' for o in orgs]) if orgs else "<span style='color:#888;font-size:0.8rem;'>None detected</span>"}
                        </div>""", unsafe_allow_html=True)

                # User news analysis
                st.markdown('<div class="sec-divider">Analyze News About This Stock</div>', unsafe_allow_html=True)
                news_input = st.text_area(f"Paste any news article about {ticker} for instant NLP analysis:",
                                         height=120, key="news_in",
                                         placeholder=f"Paste a news headline or article about {ticker}...")
                if news_input:
                    pol, sub, lbl = get_sentiment(news_input)
                    ents2  = get_ner_entities(news_input)
                    kws2   = [k for k,_ in get_tfidf_keywords(news_input, 6)]
                    sc2    = "#4caf7d" if lbl=="Positive" else ("#e85020" if lbl=="Negative" else "#888")
                    st.markdown(f"""<div class="nlp-card" style="border-color:{sc2};box-shadow:4px 4px 0 {sc2};">
                    <strong>Sentiment:</strong> <span style="color:{sc2};font-weight:700;font-size:1.1rem;">{lbl}</span>
                    &nbsp; Polarity: <span style="font-family:IBM Plex Mono,monospace;font-weight:600;">{pol:+.3f}</span>
                    &nbsp; Subjectivity: <span style="font-family:IBM Plex Mono,monospace;">{sub:.3f}</span><br>
                    <div style="margin-top:8px;">{"".join([f'<span class="metric-pill">{k}</span>' for k in kws2])}</div>
                    <div style="font-size:0.82rem;color:#666;margin-top:8px;">
                    {"📈 Positive news — typically bullish signal for the stock." if lbl=="Positive"
                    else ("📉 Negative news — may pressure the stock price." if lbl=="Negative"
                    else "➡️ Neutral/mixed — market reaction is uncertain.")}
                    </div></div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load data for {ticker}: {e}")
