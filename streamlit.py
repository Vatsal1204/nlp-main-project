# streamlit_app.py — NLP Financial Explorer (Modern Dark UI)

import streamlit as st
import pandas as pd
import numpy as np
import re, os
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go

try:
    from groq import Groq as _Groq
    GROQ_PKG_OK = True
except Exception:
    GROQ_PKG_OK = False

try:
    import yfinance as yf
    YFINANCE_OK = True
except Exception:
    YFINANCE_OK = False

@st.cache_resource
def download_nltk():
    for pkg in ['punkt','stopwords','averaged_perceptron_tagger','maxent_ne_chunker',
                'words','wordnet','punkt_tab','averaged_perceptron_tagger_eng']:
        try: nltk.download(pkg, quiet=True)
        except: pass
download_nltk()

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(page_title="NLP Finance", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ── DESIGN SYSTEM ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── Base ── */
.stApp { background: #070b14; }
.main .block-container { padding: 2rem 2rem 4rem 2rem; max-width: 1200px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 14px !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.1) !important;
    color: #e2e8f0 !important;
}

/* ── Glass Cards ── */
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
.glass-bright {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.glass-green  { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3); border-radius:16px; padding:20px 24px; margin-bottom:16px; }
.glass-red    { background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.3);  border-radius:16px; padding:20px 24px; margin-bottom:16px; }
.glass-yellow { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3); border-radius:16px; padding:20px 24px; margin-bottom:16px; }

/* ── Stat Card ── */
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
}
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: #6366f1; line-height: 1; }
.stat-lbl { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }

/* ── Page Title ── */
.page-hero {
    padding: 8px 0 24px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 28px;
}
.page-hero h1 {
    font-size: 2.2rem; font-weight: 800; color: #f1f5f9;
    letter-spacing: -0.5px; margin: 0 0 6px 0; line-height: 1.1;
}
.page-hero .sub {
    font-size: 0.82rem; color: #475569; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 500;
}
.accent { color: #6366f1; }

/* ── Badge ── */
.badge {
    display: inline-block; background: rgba(99,102,241,0.15);
    color: #818cf8; border: 1px solid rgba(99,102,241,0.3);
    font-size: 0.68rem; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; letter-spacing: 1px; text-transform: uppercase; margin: 2px;
}
.badge-green  { background:rgba(16,185,129,0.12); color:#34d399; border-color:rgba(16,185,129,0.3); }
.badge-red    { background:rgba(239,68,68,0.12);  color:#f87171; border-color:rgba(239,68,68,0.3); }
.badge-yellow { background:rgba(245,158,11,0.12); color:#fbbf24; border-color:rgba(245,158,11,0.3); }

/* ── Token chips ── */
.chip { display:inline-block; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); color:#94a3b8; font-family:'JetBrains Mono',monospace; font-size:0.72rem; padding:3px 9px; border-radius:6px; margin:2px; }
.chip-red    { background:rgba(239,68,68,0.1);  border-color:rgba(239,68,68,0.3);  color:#f87171; text-decoration:line-through; }
.chip-green  { background:rgba(16,185,129,0.1); border-color:rgba(16,185,129,0.3); color:#34d399; }
.chip-blue   { background:rgba(99,102,241,0.1); border-color:rgba(99,102,241,0.3); color:#818cf8; }

/* ── Section label ── */
.sec-label {
    font-size: 0.7rem; font-weight: 700; color: #6366f1;
    text-transform: uppercase; letter-spacing: 2px;
    padding-bottom: 8px; margin: 24px 0 14px 0;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

/* ── Sentence rows ── */
.sent-row {
    background: rgba(255,255,255,0.02); border-radius: 8px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.86rem;
    color: #cbd5e1; border-left: 3px solid transparent;
    display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px;
    margin: 8px 0; font-size: 0.88rem; max-width: 75%; margin-left: auto;
}
.bubble-bot {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    color: #cbd5e1; padding: 12px 18px; border-radius: 18px 18px 18px 4px;
    margin: 8px 0; font-size: 0.88rem; max-width: 82%; line-height: 1.6;
}

/* ── NER highlight ── */
.ner-pill { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.73rem; font-weight:600; margin:2px; font-family:'JetBrains Mono',monospace; }

/* ── Progress bar ── */
.pbar-wrap { background:rgba(255,255,255,0.06); border-radius:100px; height:8px; margin:8px 0; overflow:hidden; }
.pbar-fill  { height:100%; border-radius:100px; transition:width 0.5s; }

/* ── Streamlit overrides ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e2e8f0 !important; border-radius: 10px !important; }
.stButton > button { border-radius: 10px !important; font-weight: 600 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #4f46e5, #6366f1) !important; border: none !important; }
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:12px; }
.stExpander { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
label, .stRadio label, .stSelectbox label, .stTextInput label, .stTextArea label, .stSlider label { color: #64748b !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
p, li { color: #94a3b8; }

/* ════════════════════════════════════
   ANIMATIONS
   ════════════════════════════════════ */

/* Keyframes */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(22px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes slideInLeft {
  from { opacity: 0; transform: translateX(-24px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
  from { opacity: 0; transform: translateX(24px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.88); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 0px rgba(99,102,241,0); }
  50%       { box-shadow: 0 0 22px rgba(99,102,241,0.35); }
}
@keyframes borderGlow {
  0%, 100% { border-color: rgba(99,102,241,0.15); }
  50%       { border-color: rgba(99,102,241,0.55); }
}
@keyframes countUp {
  from { opacity: 0; transform: translateY(10px) scale(0.9); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
@keyframes typingDot {
  0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
  40%            { transform: scale(1);   opacity: 1; }
}
@keyframes orbitSpin {
  from { transform: rotate(0deg) translateX(18px) rotate(0deg); }
  to   { transform: rotate(360deg) translateX(18px) rotate(-360deg); }
}
@keyframes floatY {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-6px); }
}
@keyframes ripple {
  0%   { transform: scale(0.8); opacity: 1; }
  100% { transform: scale(2.4); opacity: 0; }
}
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes scanline {
  0%   { top: -2px; }
  100% { top: 100%; }
}
@keyframes particleFly {
  0%   { transform: translateY(0) translateX(0) scale(1); opacity: 1; }
  100% { transform: translateY(-60px) translateX(20px) scale(0); opacity: 0; }
}
@keyframes borderTrail {
  0%   { clip-path: inset(0 100% 98% 0); }
  25%  { clip-path: inset(0 0 98% 0); }
  50%  { clip-path: inset(0 0 0 98%); }
  75%  { clip-path: inset(98% 0 0 0); }
  100% { clip-path: inset(0 100% 0 0); }
}
@keyframes neonFlicker {
  0%,19%,21%,23%,25%,54%,56%,100% { opacity: 1; text-shadow: 0 0 8px #818cf8, 0 0 20px #6366f1; }
  20%,24%,55% { opacity: 0.7; text-shadow: none; }
}

/* ── Page enters ── */
.page-hero      { animation: fadeSlideUp 0.5s ease both; }
.glass          { animation: fadeSlideUp 0.45s ease both; }
.glass-bright   { animation: fadeSlideUp 0.45s ease both; }
.glass-green    { animation: fadeSlideUp 0.45s ease both; }
.stat-card      { animation: scaleIn 0.4s ease both; }
.sec-label      { animation: fadeIn 0.4s ease both; }
.sent-row       { animation: slideInLeft 0.4s ease both; }
.bubble-user    { animation: slideInRight 0.35s ease both; }
.bubble-bot     { animation: slideInLeft  0.35s ease both; }
.ner-pill       { animation: scaleIn 0.3s ease both; }
.chip           { animation: fadeIn 0.3s ease both; }
.badge          { animation: scaleIn 0.25s ease both; }

/* stagger children */
.glass:nth-child(1) { animation-delay: 0.0s; }
.glass:nth-child(2) { animation-delay: 0.07s; }
.glass:nth-child(3) { animation-delay: 0.14s; }
.glass:nth-child(4) { animation-delay: 0.21s; }
.stat-card:nth-child(1) { animation-delay: 0.0s; }
.stat-card:nth-child(2) { animation-delay: 0.06s; }
.stat-card:nth-child(3) { animation-delay: 0.12s; }
.stat-card:nth-child(4) { animation-delay: 0.18s; }
.stat-card:nth-child(5) { animation-delay: 0.24s; }
.sent-row:nth-child(1)  { animation-delay: 0.00s; }
.sent-row:nth-child(2)  { animation-delay: 0.05s; }
.sent-row:nth-child(3)  { animation-delay: 0.10s; }
.sent-row:nth-child(4)  { animation-delay: 0.15s; }
.sent-row:nth-child(5)  { animation-delay: 0.20s; }

/* ── Hover effects ── */
.glass:hover {
    border-color: rgba(99,102,241,0.3) !important;
    transform: translateY(-2px);
    transition: all 0.25s ease;
    box-shadow: 0 8px 32px rgba(99,102,241,0.1);
}
.stat-card:hover {
    animation: glowPulse 1.5s ease infinite;
    border-color: rgba(99,102,241,0.4) !important;
    transform: translateY(-3px) scale(1.02);
    transition: all 0.2s ease;
}
.chip:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #818cf8 !important;
    transform: translateY(-1px) scale(1.05);
    transition: all 0.15s ease;
    cursor: default;
}
.badge:hover {
    transform: scale(1.08);
    transition: all 0.15s ease;
}
.ner-pill:hover {
    transform: scale(1.1) translateY(-2px);
    transition: all 0.15s ease;
    cursor: default;
}
.sent-row:hover {
    background: rgba(255,255,255,0.04) !important;
    transform: translateX(4px);
    transition: all 0.2s ease;
}
.bubble-bot:hover, .bubble-user:hover {
    transform: scale(1.01);
    transition: all 0.15s ease;
}

/* ── Animated gradient background on hero ── */
.page-hero h1 .accent {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #6366f1);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
}

/* ── Stat value pop ── */
.stat-val {
    animation: countUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}

/* ── Primary button pulse ── */
.stButton > button[kind="primary"] {
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.45) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(99,102,241,0.5) !important;
    color: #818cf8 !important;
    transform: translateY(-1px) !important;
    transition: all 0.15s ease !important;
}

/* ── Sidebar logo float ── */
.sidebar-logo-icon {
    animation: floatY 3s ease-in-out infinite;
    display: inline-block;
}

/* ── Animated border on glass-bright ── */
.glass-bright {
    position: relative;
    overflow: hidden;
}
.glass-bright::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px; right: -1px; bottom: -1px;
    border-radius: 16px;
    background: linear-gradient(90deg, #4f46e5, #8b5cf6, #ec4899, #4f46e5);
    background-size: 300% 100%;
    animation: gradientShift 3s ease infinite;
    z-index: -1;
    opacity: 0.4;
}

/* ── Shimmer skeleton loader ── */
.shimmer {
    background: linear-gradient(90deg, rgba(255,255,255,0.02) 25%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.02) 75%);
    background-size: 400px 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 8px;
    height: 16px;
    margin: 6px 0;
}

/* ── Typing indicator ── */
.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #818cf8;
    border-radius: 50%;
    margin: 0 2px;
}
.typing-dot:nth-child(1) { animation: typingDot 1.2s ease infinite 0.0s; }
.typing-dot:nth-child(2) { animation: typingDot 1.2s ease infinite 0.2s; }
.typing-dot:nth-child(3) { animation: typingDot 1.2s ease infinite 0.4s; }

/* ── Similarity score ring ── */
.score-ring {
    animation: scaleIn 0.6s cubic-bezier(0.34,1.56,0.64,1) both, glowPulse 2.5s ease 0.8s infinite;
}

/* ── NER highlighted text glow ── */
mark {
    transition: all 0.2s ease;
}
mark:hover {
    filter: brightness(1.3);
    transform: scale(1.02);
    display: inline-block;
}

/* ── Pipeline step hover ── */
.pipeline-step {
    transition: all 0.2s ease;
    cursor: default;
}
.pipeline-step:hover {
    transform: translateY(-4px) scale(1.05);
    box-shadow: 0 8px 24px rgba(99,102,241,0.3);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.6); }

/* ── Input focus glow ── */
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    transition: all 0.2s ease !important;
}

/* ── Candlestick chart container ── */
.js-plotly-plot {
    animation: fadeSlideUp 0.5s ease both;
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ── GROQ CLIENT ──────────────────────────────
def _get_secret(name):
    val = os.environ.get(name, "")
    if not val:
        try: val = st.secrets[name]
        except: pass
    return (val or "").strip()

def ai_chat(messages, system="You are a helpful financial analyst.", max_tokens=700):
    key = _get_secret("GROQ_API_KEY")
    if not key:
        return "⚠️ Add GROQ_API_KEY to Streamlit Secrets. Get free key at console.groq.com"
    if not GROQ_PKG_OK:
        return "⚠️ groq package missing from requirements.txt"
    try:
        client = _Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":system}] + messages,
            max_tokens=max_tokens, temperature=0.4)
        return resp.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "429" in err: return "⚠️ Rate limit. Wait 30s and try again."
        if "401" in err: return "⚠️ Invalid API key. Check Streamlit Secrets."
        return f"⚠️ Error: {err}"

groq_chat  = ai_chat
groq_client = bool(_get_secret("GROQ_API_KEY"))

# ── SAMPLES ──────────────────────────────────
SAMPLE_TEXTS = {
    "Apple Earnings (Positive)": "Apple Inc. reported record-breaking quarterly earnings today, surpassing analyst expectations by a significant margin. The company posted revenue of $119.6 billion, up 9% year-over-year, driven by strong iPhone 15 sales and growth in Services. CEO Tim Cook expressed confidence in Apple's innovation pipeline, highlighting upcoming AI features in iOS. The stock surged 4.2% in after-hours trading following the announcement. Investors are optimistic about the company's expansion into augmented reality and financial services.",
    "Bank Crisis (Negative)":    "Silicon Valley Bank collapsed today in the largest U.S. bank failure since the 2008 financial crisis. Federal regulators seized the bank after a catastrophic run on deposits wiped out $42 billion in a single day. The collapse sent shockwaves through the tech startup ecosystem, with thousands of companies unable to access payroll funds. Treasury Secretary Janet Yellen held emergency meetings with banking regulators. Shares of other regional banks plummeted as contagion fears spread across markets.",
    "Tesla Report (Neutral)":    "Tesla delivered 484,507 vehicles in the third quarter, slightly below analyst expectations of 490,000 units. The electric vehicle maker maintained its annual delivery target despite increased competition from Chinese manufacturers including BYD. Production at the Shanghai Gigafactory continues at full capacity. Elon Musk commented on the competitive landscape during an earnings call. Analysts have mixed views on Tesla's near-term growth prospects given macroeconomic headwinds.",
    "Fed Rate Decision":         "The Federal Reserve raised interest rates by 25 basis points today, bringing the federal funds rate to a 22-year high of 5.25%. Chairman Jerome Powell signaled the possibility of additional hikes if inflation remains above the 2% target. Bond yields rose sharply following the announcement, with the 10-year Treasury yield climbing to 4.8%. Markets reacted with the S&P 500 dropping 1.2% and technology stocks bearing the brunt of the selloff. Mortgage rates are expected to remain elevated, dampening the housing market.",
}

# ── NLP UTILS ────────────────────────────────
def get_tokens(text):
    try: return word_tokenize(text.lower())
    except: return text.lower().split()

def get_sentences(text):
    try: return sent_tokenize(text)
    except: return [s.strip() for s in text.split('.') if len(s.strip())>10]

def get_sw():
    try: return set(stopwords.words('english'))
    except: return {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were'}

def get_sentiment(text):
    b = TextBlob(text)
    p, s = b.sentiment.polarity, b.sentiment.subjectivity
    return p, s, ("Positive" if p>0.1 else ("Negative" if p<-0.1 else "Neutral"))

def get_ner(text):
    ents = {"ORG":[],"GPE":[],"PERSON":[],"MONEY":[],"PCT":[]}
    ents["MONEY"] = list(set(re.findall(r'\$[\d,\.]+\s*(?:billion|million|trillion)?|\d+(?:\.\d+)?\s*(?:billion|million)\s*dollars?', text, re.I)))
    ents["PCT"]   = list(set(re.findall(r'\d+(?:\.\d+)?%', text)))
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    ORGS    = ['Apple','Tesla','Google','Microsoft','Amazon','Meta','Netflix','Nvidia','Bank','Federal Reserve','FDIC','Treasury','Inc','Corp','BYD','Gigafactory','Silicon Valley Bank']
    GPES    = ['U.S.','United States','China','Europe','Shanghai','New York','Washington']
    PERSONS = ['Powell','Musk','Cook','Yellen','Biden','Jerome','Tim','Elon','Janet']
    for w in caps:
        if   any(o.lower() in w.lower() for o in ORGS)    and w not in ents["ORG"]:    ents["ORG"].append(w)
        elif any(g.lower() in w.lower() for g in GPES)    and w not in ents["GPE"]:    ents["GPE"].append(w)
        elif any(p.lower() in w.lower() for p in PERSONS) and w not in ents["PERSON"]: ents["PERSON"].append(w)
    for k in ents: ents[k] = list(dict.fromkeys(ents[k]))[:8]
    return ents

def get_tfidf(text, n=15):
    sents = get_sentences(text)
    if len(sents)<2: sents=[text,text]
    try:
        v = TfidfVectorizer(stop_words='english', max_features=100)
        m = v.fit_transform(sents)
        sc = np.array(m.sum(axis=0)).flatten()
        idx = np.argsort(-sc)[:n]
        return [(v.get_feature_names_out()[i], float(sc[i])) for i in idx]
    except: return []

def sim_matrix(texts):
    try:
        v = TfidfVectorizer(stop_words='english')
        return cosine_similarity(v.fit_transform(texts))
    except: return None

# ── DARK PLOTLY THEME ────────────────────────
PLOT_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.02)',
    font=dict(family="JetBrains Mono", color="#64748b"),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
    margin=dict(l=0, r=0, t=40, b=0),
)

def dark_fig(fig, h=300):
    fig.update_layout(height=h, **PLOT_DARK)
    return fig

# ── SHARED INPUT WIDGET ──────────────────────
def text_input_widget(key, default="Apple Earnings (Positive)"):
    c1, c2 = st.columns([1,3])
    with c1:
        s = st.selectbox("Sample", list(SAMPLE_TEXTS.keys()),
                         index=list(SAMPLE_TEXTS.keys()).index(default), key=f"{key}_s")
    with c2:
        t = st.text_area("Or paste your own text", value=SAMPLE_TEXTS.get(s,""),
                         height=110, key=f"{key}_t",
                         placeholder="Paste any financial news, earnings report, or market commentary...")
    return t.strip()

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 16px 20px;border-bottom:1px solid rgba(99,102,241,0.15);margin-bottom:16px;'>
      <div style='display:flex;align-items:center;gap:10px;'>
        <div class="sidebar-logo-icon" style='width:36px;height:36px;background:linear-gradient(135deg,#4f46e5,#818cf8);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;'>⚡</div>
        <div>
          <div style='font-size:0.95rem;font-weight:800;color:#f1f5f9 !important;'>NLP Finance</div>
          <div style='font-size:0.65rem;color:#475569 !important;letter-spacing:1.5px;text-transform:uppercase;'>Explorer</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Overview",
        "✂️  Preprocessing",
        "😊  Sentiment",
        "🏷️  Entities (NER)",
        "📊  Frequency & TF-IDF",
        "🔍  Similarity",
        "📝  Summarization",
        "🤖  AI Chatbot",
        "📈  Stock + NLP",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:16px 0;'></div>", unsafe_allow_html=True)
    api_ok = groq_client
    dot = "🟢" if api_ok else "🔴"
    st.markdown(f"<div style='font-size:0.72rem;color:#475569;padding:0 4px;'>{dot} Groq API {'Connected' if api_ok else 'Not configured'}</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#334155;margin-top:6px;padding:0 4px;'>NLTK · TextBlob · sklearn · yfinance</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div class="page-hero">
      <h1>NLP Financial <span class="accent">Explorer</span></h1>
      <div class="sub">Natural Language Processing · All Topics · Interactive</div>
    </div>
    """, unsafe_allow_html=True)

    topics = [
        ("✂️","Preprocessing",   "Tokenize → Stopwords → Stem → Lemmatize → POS",  "#6366f1"),
        ("😊","Sentiment",        "Polarity · Subjectivity · Per-sentence scoring",   "#10b981"),
        ("🏷️","Entities (NER)",  "ORG · PERSON · MONEY · GPE · Highlighted text",   "#f59e0b"),
        ("📊","Freq & TF-IDF",    "Bag of Words · TF-IDF · Word Cloud · N-grams",    "#3b82f6"),
        ("🔍","Similarity",       "Cosine Similarity · Jaccard · Heatmap",            "#8b5cf6"),
        ("📝","Summarization",    "Extractive TF-IDF + Abstractive AI (Groq)",        "#ec4899"),
        ("🤖","AI Chatbot",       "LLaMA 3.3 70B · Context-aware · Finance Q&A",     "#6366f1"),
        ("📈","Stock + NLP",      "yfinance · Live prices · News sentiment",          "#10b981"),
    ]
    for i in range(0, len(topics), 4):
        cols = st.columns(4)
        for col, t in zip(cols, topics[i:i+4]):
            icon, name, desc, color = t
            with col:
                st.markdown(f"""<div class="glass" style="border-color:rgba(99,102,241,0.15);min-height:130px;cursor:pointer;transition:all 0.2s;">
                <div style="font-size:1.6rem;margin-bottom:8px;">{icon}</div>
                <div style="font-size:0.9rem;font-weight:700;color:#f1f5f9;margin-bottom:6px;">{name}</div>
                <div style="font-size:0.73rem;color:#475569;line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">The NLP Pipeline</div>', unsafe_allow_html=True)
    steps = ["Raw Text","Tokenize","Clean","Vectorize","Model","Insights"]
    cols = st.columns(len(steps))
    for i,(col,s) in enumerate(zip(cols,steps)):
        bg = "linear-gradient(135deg,#4f46e5,#6366f1)" if i==0 else ("rgba(99,102,241,0.15)" if i==len(steps)-1 else "rgba(255,255,255,0.03)")
        tc = "white" if i==0 else ("#818cf8" if i==len(steps)-1 else "#64748b")
        with col:
            st.markdown(f"""<div class="pipeline-step" style="background:{bg};border:1px solid rgba(99,102,241,0.2);padding:12px 6px;border-radius:10px;text-align:center;">
            <div style="font-size:0.65rem;font-weight:700;color:{tc};text-transform:uppercase;letter-spacing:1px;">{i+1}</div>
            <div style="font-size:0.78rem;font-weight:600;color:{tc};margin-top:2px;">{s}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Live Demo — Instant Sentiment</div>', unsafe_allow_html=True)
    demo = st.text_input("", placeholder="Type any financial sentence... e.g. 'Apple crushed earnings expectations'", label_visibility="collapsed")
    if demo:
        p, s, l = get_sentiment(demo)
        colors = {"Positive":"#10b981","Negative":"#ef4444","Neutral":"#64748b"}
        c = colors[l]
        st.markdown(f"""<div class="glass" style="border-color:{c}33;display:flex;align-items:center;gap:20px;padding:16px 24px;">
        <div style="font-size:2rem;">{"📈" if l=="Positive" else ("📉" if l=="Negative" else "➡️")}</div>
        <div>
          <div style="font-size:1.4rem;font-weight:800;color:{c};">{l}</div>
          <div style="font-size:0.78rem;color:#475569;font-family:'JetBrains Mono',monospace;">polarity: {p:+.3f} &nbsp;·&nbsp; subjectivity: {s:.3f}</div>
        </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════
elif page == "✂️  Preprocessing":
    st.markdown('<div class="page-hero"><h1>Text <span class="accent">Preprocessing</span></h1><div class="sub">Tokenization · Stopwords · Stemming · Lemmatization · POS</div></div>', unsafe_allow_html=True)

    text = text_input_widget("prep")
    if not text: st.warning("Enter some text above."); st.stop()

    tokens = get_tokens(text); sw = get_sw()
    alpha  = [t for t in tokens if t.isalpha()]
    filt   = [t for t in alpha if t not in sw]
    stemmed = [PorterStemmer().stem(t) for t in filt]
    lemmed  = [WordNetLemmatizer().lemmatize(t) for t in filt]
    sents   = get_sentences(text)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,v,l in zip([c1,c2,c3,c4,c5],
        [len(sents),len(tokens),len(filt),len(set(filt)),round(len(filt)/max(len(sents),1),1)],
        ["Sentences","All Tokens","After Stopwords","Unique Words","Words/Sentence"]):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{v}</div><div class="stat-lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Step 1 — Tokenization</div>', unsafe_allow_html=True)
    html = "".join([f'<span class="chip">{t}</span>' for t in tokens[:60]])
    if len(tokens)>60: html += f'<span style="color:#334155;font-size:0.75rem;"> +{len(tokens)-60} more</span>'
    st.markdown(f'<div class="glass" style="padding:16px;">{html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Step 2 — Stopword Removal</div>', unsafe_allow_html=True)
    html2 = "".join([f'<span class="chip chip-red">{t}</span>' if t in sw else f'<span class="chip">{t}</span>' for t in alpha[:80]])
    pct = round((1-len(filt)/max(len(alpha),1))*100)
    st.markdown(f'<div class="glass" style="padding:16px;">{html2}<br><br><span class="badge badge-red">{pct}% removed as stopwords</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Step 3 — Stemming vs Lemmatization</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        chips = "".join([f'<span class="chip chip-green">{s}</span>' for s in stemmed[:30]])
        st.markdown(f'<div class="glass"><div style="font-weight:700;color:#f1f5f9;margin-bottom:4px;">⚡ Porter Stemmer</div><div style="font-size:0.75rem;color:#475569;margin-bottom:10px;">"running"→"run" · "studies"→"studi" · fast but crude</div>{chips}</div>', unsafe_allow_html=True)
    with c2:
        chips2 = "".join([f'<span class="chip chip-blue">{l}</span>' for l in lemmed[:30]])
        st.markdown(f'<div class="glass"><div style="font-weight:700;color:#f1f5f9;margin-bottom:4px;">✨ WordNet Lemmatizer</div><div style="font-size:0.75rem;color:#475569;margin-bottom:10px;">"better"→"good" · "studies"→"study" · dictionary-accurate</div>{chips2}</div>', unsafe_allow_html=True)

    if filt:
        st.markdown('<div class="sec-label">Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Original":filt[:20],"Stemmed":stemmed[:20],"Lemmatized":lemmed[:20]}), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-label">Step 4 — POS Tagging</div>', unsafe_allow_html=True)
    try:
        pt = word_tokenize(text[:500]); tagged = pos_tag(pt)
        pc = {"NN":"#f59e0b","VB":"#6366f1","JJ":"#10b981","RB":"#8b5cf6","NNP":"#ef4444"}
        def _pos_chip(word, tag):
            col = pc.get(tag[:2], "#475569")
            return (f'<span style="display:inline-block;margin:2px;padding:2px 8px;'
                    f'background:rgba(255,255,255,0.03);border:1px solid {col}33;'
                    f'border-radius:6px;font-family:JetBrains Mono,monospace;font-size:0.68rem;">'
                    f'<span style="color:{col};font-weight:700;">{tag}</span> '
                    f'<span style="color:#94a3b8;">{word}</span></span>')
        pos_html = "".join([_pos_chip(w,t) for w,t in tagged[:50] if w.isalpha()])
        st.markdown(f'<div class="glass" style="padding:16px;">{pos_html}</div>', unsafe_allow_html=True)
        cnt = Counter([tag[:2] for _,tag in tagged if tag[:2] in pc])
        if cnt:
            fig = px.bar(x=list(pc.keys()), y=[cnt.get(k,0) for k in pc], color=list(pc.keys()),
                        color_discrete_sequence=list(pc.values()), title="POS Distribution")
            st.plotly_chart(dark_fig(fig, 240), use_container_width=True)
    except Exception as e:
        st.info(f"POS tagging: {e}")

# ═══════════════════════════════════════════
# SENTIMENT
# ═══════════════════════════════════════════
elif page == "😊  Sentiment":
    st.markdown('<div class="page-hero"><h1>Sentiment <span class="accent">Analysis</span></h1><div class="sub">Polarity · Subjectivity · Per-sentence breakdown</div></div>', unsafe_allow_html=True)

    text = text_input_widget("sent")
    if not text: st.warning("Enter some text."); st.stop()

    pol, sub, lbl = get_sentiment(text)
    COLS = {"Positive":"#10b981","Negative":"#ef4444","Neutral":"#64748b"}
    c = COLS[lbl]
    icon = "📈" if lbl=="Positive" else ("📉" if lbl=="Negative" else "➡️")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="glass" style="border-color:{c}33;text-align:center;padding:28px 20px;"><div style="font-size:2.5rem;">{icon}</div><div style="font-size:1.8rem;font-weight:800;color:{c};margin:8px 0;">{lbl}</div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;">Overall Sentiment</div></div>', unsafe_allow_html=True)
    with c2:
        pp = int((pol+1)/2*100)
        fc = "#10b981" if pol>0.1 else ("#ef4444" if pol<-0.1 else "#64748b")
        st.markdown(f'<div class="glass"><div style="font-family:JetBrains Mono,monospace;font-size:2rem;font-weight:700;color:#f1f5f9;">{pol:+.3f}</div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin:6px 0 10px;">Polarity  ( -1 → +1 )</div><div class="pbar-wrap"><div class="pbar-fill" style="width:{pp}%;background:{fc};"></div></div><div style="font-size:0.7rem;color:#334155;display:flex;justify-content:space-between;margin-top:4px;"><span>Negative</span><span>Positive</span></div></div>', unsafe_allow_html=True)
    with c3:
        sp = int(sub*100)
        st.markdown(f'<div class="glass"><div style="font-family:JetBrains Mono,monospace;font-size:2rem;font-weight:700;color:#f1f5f9;">{sub:.3f}</div><div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin:6px 0 10px;">Subjectivity  ( 0 → 1 )</div><div class="pbar-wrap"><div class="pbar-fill" style="width:{sp}%;background:#6366f1;"></div></div><div style="font-size:0.7rem;color:#334155;display:flex;justify-content:space-between;margin-top:4px;"><span>Objective</span><span>Subjective</span></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Sentence Breakdown</div>', unsafe_allow_html=True)
    sents = get_sentences(text)
    rows = []
    for s in sents:
        if len(s.split())<4: continue
        p,sb,l = get_sentiment(s)
        rows.append({"Sentence":s,"Polarity":round(p,3),"Label":l})
    for r in rows:
        bc = COLS[r["Label"]]
        si = "📈" if r["Label"]=="Positive" else ("📉" if r["Label"]=="Negative" else "➡️")
        st.markdown(f'<div class="sent-row" style="border-left-color:{bc};"><span style="color:#94a3b8;flex:1;">{si} {r["Sentence"]}</span><span style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:{bc};font-weight:600;white-space:nowrap;">{r["Polarity"]:+.3f}</span></div>', unsafe_allow_html=True)

    if rows:
        df_s = pd.DataFrame(rows)
        c1,c2 = st.columns(2)
        with c1:
            lc = Counter([r["Label"] for r in rows])
            fig = px.pie(values=list(lc.values()), names=list(lc.keys()), hole=0.55,
                        color=list(lc.keys()), color_discrete_map={"Positive":"#10b981","Negative":"#ef4444","Neutral":"#475569"})
            fig.update_traces(textfont_color="white")
            st.plotly_chart(dark_fig(fig,260), use_container_width=True)
        with c2:
            fig2 = px.bar(df_s, y="Polarity", color="Label",
                         color_discrete_map={"Positive":"#10b981","Negative":"#ef4444","Neutral":"#475569"})
            fig2.add_hline(y=0, line_color="rgba(255,255,255,0.1)", line_width=1)
            st.plotly_chart(dark_fig(fig2,260), use_container_width=True)

    st.markdown('<div class="sec-label">Compare All Samples</div>', unsafe_allow_html=True)
    cmp = [{"Text":n,"Polarity":round(get_sentiment(t)[0],3),"Sentiment":get_sentiment(t)[2]} for n,t in SAMPLE_TEXTS.items()]
    fig3 = px.bar(pd.DataFrame(cmp), x="Text", y="Polarity", color="Sentiment",
                 color_discrete_map={"Positive":"#10b981","Negative":"#ef4444","Neutral":"#475569"})
    fig3.add_hline(y=0, line_color="rgba(255,255,255,0.1)", line_width=1)
    st.plotly_chart(dark_fig(fig3,280), use_container_width=True)

# ═══════════════════════════════════════════
# NER
# ═══════════════════════════════════════════
elif page == "🏷️  Entities (NER)":
    st.markdown('<div class="page-hero"><h1>Named Entity <span class="accent">Recognition</span></h1><div class="sub">ORG · GPE · PERSON · MONEY · PCT</div></div>', unsafe_allow_html=True)

    text = text_input_widget("ner")
    if not text: st.warning("Enter some text."); st.stop()

    ents = get_ner(text)
    TAG = {
        "ORG":    ("🏢","rgba(245,158,11,0.12)","#f59e0b","Organization"),
        "GPE":    ("🌍","rgba(59,130,246,0.12)", "#3b82f6","Location"),
        "PERSON": ("👤","rgba(239,68,68,0.12)",  "#ef4444","Person"),
        "MONEY":  ("💰","rgba(16,185,129,0.12)", "#10b981","Money"),
        "PCT":    ("📌","rgba(139,92,246,0.12)", "#8b5cf6","Percentage"),
    }
    has = any(ents[k] for k in ents)
    if not has:
        st.markdown('<div class="glass"><div style="color:#475569;text-align:center;padding:20px;">No entities detected — try the Apple Earnings or Bank Crisis sample.</div></div>', unsafe_allow_html=True)

    cols = st.columns(len([k for k in TAG if ents.get(k)]) or 1)
    ci = 0
    for etype,(icon,bg,color,label) in TAG.items():
        el = ents.get(etype,[])
        if not el: continue
        with cols[ci]:
            pills = "".join([f'<span class="ner-pill" style="background:{bg};color:{color};border:1px solid {color}33;animation:scaleIn 0.3s ease {0.05*j:.2f}s both;">{e}</span>' for j,e in enumerate(el)])
            st.markdown(f'<div class="glass" style="border-color:{color}22;animation:fadeSlideUp 0.4s ease {0.1*ci:.1f}s both;"><div style="font-size:0.68rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">{icon} {label}</div>{pills}</div>', unsafe_allow_html=True)
        ci += 1

    if has:
        st.markdown('<div class="sec-label">Highlighted Text</div>', unsafe_allow_html=True)
        highlighted = text
        for etype,el in ents.items():
            _,bg,color,_ = TAG[etype]
            for e in el:
                highlighted = highlighted.replace(e, f'<mark style="background:{bg};color:{color};padding:1px 6px;border-radius:4px;font-weight:600;">{e}<sup style="font-size:0.55rem;opacity:0.7;"> {etype}</sup></mark>')
        st.markdown(f'<div class="glass" style="font-size:0.88rem;color:#94a3b8;line-height:2;">{highlighted}</div>', unsafe_allow_html=True)

        ec = {TAG[k][3]:len(v) for k,v in ents.items() if v}
        if ec:
            fig = px.bar(x=list(ec.keys()), y=list(ec.values()), color=list(ec.keys()),
                        color_discrete_sequence=["#f59e0b","#3b82f6","#ef4444","#10b981","#8b5cf6"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(dark_fig(fig,220), use_container_width=True)

# ═══════════════════════════════════════════
# WORD FREQUENCY & TF-IDF
# ═══════════════════════════════════════════
elif page == "📊  Frequency & TF-IDF":
    st.markdown('<div class="page-hero"><h1>Frequency & <span class="accent">TF-IDF</span></h1><div class="sub">Word Counts · TF-IDF Keywords · Word Cloud · N-grams</div></div>', unsafe_allow_html=True)

    text = text_input_widget("freq")
    if not text: st.warning("Enter some text."); st.stop()

    tokens = get_tokens(text); sw = get_sw()
    filt   = [t for t in tokens if t.isalpha() and t not in sw and len(t)>2]
    freq   = Counter(filt)
    top20  = freq.most_common(20)
    kws    = get_tfidf(text, 20)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-label">Word Frequency</div>', unsafe_allow_html=True)
        if top20:
            df_f = pd.DataFrame(top20, columns=["Word","Count"])
            fig = px.bar(df_f, x="Count", y="Word", orientation='h',
                        color="Count", color_continuous_scale=["#1e1b4b","#4f46e5","#818cf8"])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(dark_fig(fig,440), use_container_width=True)
    with c2:
        st.markdown('<div class="sec-label">TF-IDF Keywords</div>', unsafe_allow_html=True)
        if kws:
            df_t = pd.DataFrame(kws, columns=["Keyword","Score"])
            fig2 = px.bar(df_t, x="Score", y="Keyword", orientation='h',
                         color="Score", color_continuous_scale=["#064e3b","#10b981","#6ee7b7"])
            fig2.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(dark_fig(fig2,440), use_container_width=True)

    st.markdown('<div class="sec-label">Word Cloud</div>', unsafe_allow_html=True)
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(width=1000, height=300, background_color="#0d1117",
                       colormap="cool", max_words=70).generate(" ".join(filt))
        fig_wc, ax = plt.subplots(figsize=(12,3.5))
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        fig_wc.patch.set_facecolor('#0d1117')
        st.pyplot(fig_wc)
    except ImportError:
        mc = top20[0][1] if top20 else 1
        cloud = "".join([f'<span style="font-size:{0.7+(c/mc)*1.5:.1f}rem;opacity:{0.35+(c/mc)*0.65:.2f};color:#818cf8;margin:4px 6px;display:inline-block;font-weight:700;">{w}</span>' for w,c in freq.most_common(40)])
        st.markdown(f'<div class="glass" style="min-height:100px;text-align:center;padding:24px;">{cloud}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">N-grams</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    for col, ng, title, cscale in [(c1,(2,2),"Bigrams",["#1e1b4b","#6366f1"]), (c2,(3,3),"Trigrams",["#064e3b","#10b981"])]:
        with col:
            try:
                v = CountVectorizer(ngram_range=ng, stop_words='english', max_features=10)
                v.fit_transform([text])
                d = dict(zip(v.get_feature_names_out(), v.transform([text]).toarray()[0]))
                df_ng = pd.DataFrame(sorted(d.items(),key=lambda x:-x[1])[:10], columns=["Phrase","Count"])
                fig_ng = px.bar(df_ng, x="Count", y="Phrase", orientation='h', title=title,
                               color="Count", color_continuous_scale=cscale)
                fig_ng.update_layout(coloraxis_showscale=False, showlegend=False)
                st.plotly_chart(dark_fig(fig_ng,280), use_container_width=True)
            except: st.info("Add more text for n-grams.")

# ═══════════════════════════════════════════
# SIMILARITY
# ═══════════════════════════════════════════
elif page == "🔍  Similarity":
    st.markdown('<div class="page-hero"><h1>Text <span class="accent">Similarity</span></h1><div class="sub">Cosine Similarity · TF-IDF Vectors · Jaccard Index</div></div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-label">Text A</div>', unsafe_allow_html=True)
        sa = st.selectbox("", list(SAMPLE_TEXTS.keys()), index=0, key="sia", label_visibility="collapsed")
        ta = st.text_area("", value=SAMPLE_TEXTS[sa], height=140, key="ta", label_visibility="collapsed")
    with c2:
        st.markdown('<div class="sec-label">Text B</div>', unsafe_allow_html=True)
        sb = st.selectbox("", list(SAMPLE_TEXTS.keys()), index=1, key="sib", label_visibility="collapsed")
        tb = st.text_area("", value=SAMPLE_TEXTS[sb], height=140, key="tb", label_visibility="collapsed")

    if ta and tb:
        sm = sim_matrix([ta, tb])
        if sm is not None:
            score = sm[0][1]; pct = int(score*100)
            c = "#10b981" if pct>50 else ("#f59e0b" if pct>20 else "#ef4444")
            label = "High similarity" if pct>50 else ("Moderate overlap" if pct>20 else "Low similarity")

            st.markdown(f'<div class="glass score-ring" style="border-color:{c}33;text-align:center;padding:32px;"><div style="font-family:JetBrains Mono,monospace;font-size:4rem;font-weight:800;color:{c};text-shadow:0 0 30px {c}66;">{pct}%</div><div style="color:#475569;font-size:0.85rem;margin-top:4px;">Cosine Similarity · {label}</div></div>', unsafe_allow_html=True)

            sw = get_sw()
            wa = {t for t in get_tokens(ta) if t.isalpha() and t not in sw}
            wb = {t for t in get_tokens(tb) if t.isalpha() and t not in sw}
            common = wa&wb; jac = len(common)/len(wa|wb) if (wa|wb) else 0

            col1,col2,col3 = st.columns(3)
            for col, words, lbl, color in [(col1,list(common)[:14],"Common","#10b981"),(col2,list(wa-wb)[:12],"Only in A","#6366f1"),(col3,list(wb-wa)[:12],"Only in B","#f59e0b")]:
                with col:
                    pills = "".join([f'<span style="display:inline-block;background:{color}15;color:{color};border:1px solid {color}33;font-size:0.7rem;padding:2px 8px;border-radius:20px;margin:2px;font-family:JetBrains Mono,monospace;">{w}</span>' for w in words])
                    st.markdown(f'<div class="glass" style="min-height:100px;border-color:{color}22;"><div style="font-size:0.68rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">{lbl}</div>{pills}</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="badge">Jaccard: {jac:.3f}</span>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">All Samples Heatmap</div>', unsafe_allow_html=True)
    names = list(SAMPLE_TEXTS.keys()); txts = list(SAMPLE_TEXTS.values())
    sm_all = sim_matrix(txts)
    if sm_all is not None:
        fig = px.imshow(sm_all, x=names, y=names, color_continuous_scale=["#0f172a","#4f46e5","#818cf8"],
                       zmin=0, zmax=1, text_auto=".2f")
        st.plotly_chart(dark_fig(fig,360), use_container_width=True)

# ═══════════════════════════════════════════
# SUMMARIZATION
# ═══════════════════════════════════════════
elif page == "📝  Summarization":
    st.markdown('<div class="page-hero"><h1>Text <span class="accent">Summarization</span></h1><div class="sub">Extractive TF-IDF · Abstractive AI (Groq LLaMA 3)</div></div>', unsafe_allow_html=True)

    text = text_input_widget("summ", "Bank Crisis (Negative)")
    if not text: st.warning("Enter some text."); st.stop()

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-label">📌 Extractive — TF-IDF (no AI)</div>', unsafe_allow_html=True)
        n_s = st.slider("Sentences to extract", 1, 5, 2, key="es_n")
        sents = get_sentences(text)
        if len(sents) <= n_s:
            st.markdown(f'<div class="glass-bright" style="color:#c7d2fe;line-height:1.8;">{text}</div>', unsafe_allow_html=True)
        else:
            try:
                tv = TfidfVectorizer(stop_words='english')
                tm = tv.fit_transform(sents)
                scores = np.array(tm.sum(axis=1)).flatten()
                idxs = sorted(np.argsort(-scores)[:n_s])
                summary = " ".join([sents[i] for i in idxs])
                comp = round((1-len(summary.split())/max(len(text.split()),1))*100)
                st.markdown(f'<div class="glass-bright" style="color:#c7d2fe;line-height:1.8;">{summary}</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="badge badge-green">⬇ {comp}% compression</span>', unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

    with c2:
        st.markdown('<div class="sec-label">✨ Abstractive — Groq LLaMA 3</div>', unsafe_allow_html=True)
        style = st.selectbox("Style", ["Bullet points (investor)","One paragraph","ELI5 — simple language","Risk & downside focus"], key="ss")
        if st.button("✨ Generate AI Summary", type="primary"):
            P = {
                "Bullet points (investor)":"Summarize in 4-5 bullet points for an investor. Include key numbers.",
                "One paragraph":"Summarize in one clear, concise paragraph.",
                "ELI5 — simple language":"Explain this financial text simply for a 12-year-old.",
                "Risk & downside focus":"Summarize focusing on risks and negative implications for investors."
            }
            gen_placeholder = st.empty()
            gen_placeholder.markdown('<div style="padding:12px;"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span><span style="font-size:0.75rem;color:#475569;margin-left:8px;">Generating summary...</span></div>', unsafe_allow_html=True)
            result = groq_chat([{"role":"user","content":f"{P[style]}\n\nText:\n{text}"}],
                               system="You are a concise financial analyst. Be direct.")
            gen_placeholder.empty()
            st.markdown(f'<div class="glass-bright" style="color:#c7d2fe;line-height:1.8;white-space:pre-wrap;">{result}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass" style="text-align:center;padding:32px;color:#334155;">Click to generate AI summary ✨</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# CHATBOT
# ═══════════════════════════════════════════
elif page == "🤖  AI Chatbot":
    st.markdown('<div class="page-hero"><h1>Financial <span class="accent">AI Chatbot</span></h1><div class="sub">Groq LLaMA 3.3 · 70B · Context-Aware Q&A</div></div>', unsafe_allow_html=True)

    if not groq_client:
        st.markdown("""<div class="glass-bright" style="border-color:rgba(239,68,68,0.4);">
        <div style="color:#f87171;font-weight:800;font-size:1.1rem;margin-bottom:14px;">⚠️ Groq API Key Required</div>
        <div style="color:#94a3b8;line-height:2.2;font-size:0.88rem;">
        <b>Step 1</b> → Go to <a href="https://console.groq.com" target="_blank" style="color:#818cf8;">console.groq.com</a> → Sign up free<br>
        <b>Step 2</b> → API Keys → Create API Key → Copy it<br>
        <b>Step 3</b> → Streamlit app → Settings → Secrets → add:<br>
        <code style="background:rgba(255,255,255,0.05);padding:4px 12px;border-radius:6px;color:#818cf8;">GROQ_API_KEY = "gsk_..."</code><br>
        <b>Step 4</b> → Save → Reboot app
        </div>
        <div style="margin-top:12px;font-size:0.75rem;color:#334155;">✅ Free · Works in India · 14,400 req/day · No credit card</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    with st.expander("📄 Add context document for document Q&A"):
        cx = st.selectbox("Sample context", ["None"]+list(SAMPLE_TEXTS.keys()))
        ctx = SAMPLE_TEXTS.get(cx,"") if cx!="None" else ""
        cc = st.text_area("Or paste custom context", height=80, key="ctx_c")
        if cc: ctx = cc

    st.markdown('<div class="sec-label">Quick Questions</div>', unsafe_allow_html=True)
    quick = ["What is P/E ratio?","What does a Fed rate hike do to stocks?",
             "Explain earnings per share","What is inflation?","What is a short squeeze?","How do bonds work?"]
    cols = st.columns(3)
    for i,q in enumerate(quick):
        with cols[i%3]:
            if st.button(q, key=f"qb_{i}", use_container_width=True):
                st.session_state["_pq"] = q

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        st.markdown('<div class="sec-label">Conversation</div>', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"]=="user":
                st.markdown(f'<div style="display:flex;justify-content:flex-end;"><div class="bubble-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bubble-bot">{msg["content"].replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Ask a Question</div>', unsafe_allow_html=True)
    dq = st.session_state.pop("_pq","")
    user_q = st.text_input("", value=dq, placeholder="e.g. What caused the 2008 financial crisis?", key="chat_q", label_visibility="collapsed")
    c1,c2 = st.columns([1,6])
    with c1:
        send = st.button("Send →", type="primary")
    with c2:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_q:
        sys = "You are an expert financial analyst. Answer clearly, use line breaks. Be direct and helpful."
        if ctx: sys += f"\n\nContext:\n{ctx[:2000]}"
        typing_placeholder = st.empty()
        typing_placeholder.markdown('<div style="padding:8px 0;"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span><span style="font-size:0.75rem;color:#475569;margin-left:8px;">AI is thinking...</span></div>', unsafe_allow_html=True)
        reply = groq_chat(st.session_state.chat_history[-6:]+[{"role":"user","content":user_q}], system=sys)
        typing_placeholder.empty()
        st.session_state.chat_history += [{"role":"user","content":user_q},{"role":"assistant","content":reply}]
        st.rerun()

# ═══════════════════════════════════════════
# STOCK + NLP
# ═══════════════════════════════════════════
elif page == "📈  Stock + NLP":
    st.markdown('<div class="page-hero"><h1>Stock Data + <span class="accent">NLP</span></h1><div class="sub">yfinance · Live Prices · Sentiment on News</div></div>', unsafe_allow_html=True)

    c1,c2 = st.columns([1,2])
    with c1:
        ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="TSLA, MSFT, GOOGL").upper()
        period = st.selectbox("Period", ["1mo","3mo","6mo","1y"], index=1)

    if not YFINANCE_OK:
        st.warning("yfinance not installed."); st.stop()

    if ticker:
        try:
            with st.spinner(f"Fetching {ticker}..."):
                stock = yf.Ticker(ticker)
                hist  = stock.history(period=period)
                info  = stock.info

            if hist.empty:
                st.error(f"No data for {ticker}")
            else:
                price  = hist['Close'].iloc[-1]
                chg    = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                chgpct = chg/hist['Close'].iloc[-2]*100
                up     = chg >= 0

                with c1:
                    c = "#10b981" if up else "#ef4444"
                    st.markdown(f"""<div class="glass" style="border-color:{c}33;text-align:center;padding:24px 16px;">
                    <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;">{ticker}</div>
                    <div style="font-family:JetBrains Mono,monospace;font-size:2.2rem;font-weight:800;color:#f1f5f9;">${price:.2f}</div>
                    <div style="font-family:JetBrains Mono,monospace;color:{c};font-size:1rem;font-weight:600;">{chg:+.2f} ({chgpct:+.2f}%)</div>
                    </div>""", unsafe_allow_html=True)
                    mcap = info.get("marketCap")
                    if mcap:
                        ms = f"${mcap/1e12:.2f}T" if mcap>1e12 else f"${mcap/1e9:.1f}B"
                        st.markdown(f'<span class="badge">{ms} Mkt Cap</span>', unsafe_allow_html=True)
                    sec = info.get("sector")
                    if sec: st.markdown(f'<span class="badge badge-green">{sec}</span>', unsafe_allow_html=True)

                with c2:
                    fig = go.Figure(go.Candlestick(
                        x=hist.index, open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'], name=ticker,
                        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
                        increasing_fillcolor='rgba(16,185,129,0.3)', decreasing_fillcolor='rgba(239,68,68,0.3)'))
                    fig.update_layout(xaxis_rangeslider_visible=False, title=f"{ticker} — {period}")
                    st.plotly_chart(dark_fig(fig,320), use_container_width=True)

                desc = info.get("longBusinessSummary","")
                if desc:
                    st.markdown('<div class="sec-label">NLP on Company Description</div>', unsafe_allow_html=True)
                    pol,sub,lbl = get_sentiment(desc)
                    kws = [k for k,_ in get_tfidf(desc,8)]
                    ents = get_ner(desc)
                    sc = "#10b981" if lbl=="Positive" else ("#ef4444" if lbl=="Negative" else "#64748b")
                    col1,col2,col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="glass" style="text-align:center;border-color:{sc}33;"><div style="font-size:1.3rem;font-weight:800;color:{sc};">{lbl}</div><div style="font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#475569;">polarity: {pol:+.3f}</div></div>', unsafe_allow_html=True)
                    with col2:
                        pills = "".join([f'<span class="badge">{k}</span>' for k in kws])
                        st.markdown(f'<div class="glass"><div style="font-size:0.65rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">🔑 Keywords</div>{pills}</div>', unsafe_allow_html=True)
                    with col3:
                        orgs = ents.get("ORG",[])[:5]
                        opills = "".join([f'<span class="badge badge-yellow">{o}</span>' for o in orgs]) if orgs else '<span style="color:#334155;font-size:0.8rem;">None detected</span>'
                        st.markdown(f'<div class="glass"><div style="font-size:0.65rem;font-weight:700;color:#f59e0b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">🏢 Organizations</div>{opills}</div>', unsafe_allow_html=True)

                st.markdown('<div class="sec-label">Analyze News</div>', unsafe_allow_html=True)
                news = st.text_area(f"Paste any news about {ticker}:", height=110, key="news_in",
                                    placeholder=f"Paste a news headline or article about {ticker}...")
                if news:
                    pol,sub,lbl = get_sentiment(news)
                    kws2 = [k for k,_ in get_tfidf(news,6)]
                    sc2  = "#10b981" if lbl=="Positive" else ("#ef4444" if lbl=="Negative" else "#64748b")
                    msg  = "📈 Positive signal — typically bullish for the stock." if lbl=="Positive" else ("📉 Negative signal — may pressure the price." if lbl=="Negative" else "➡️ Neutral — market impact uncertain.")
                    pills3 = "".join([f'<span class="badge">{k}</span>' for k in kws2])
                    st.markdown(f'<div class="glass" style="border-color:{sc2}33;"><div style="display:flex;align-items:center;gap:16px;margin-bottom:10px;"><div style="font-size:1.4rem;font-weight:800;color:{sc2};">{lbl}</div><div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;color:#475569;">polarity: {pol:+.3f} · subjectivity: {sub:.3f}</div></div>{pills3}<div style="font-size:0.82rem;color:#475569;margin-top:10px;">{msg}</div></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
