import streamlit as st
import pandas as pd
import sqlite3
import os
from groq import Groq

st.set_page_config(page_title="Financial NLP Query Engine", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #424242; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Financial NLP Query Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about stock data in plain English</p>', unsafe_allow_html=True)

# Create in-memory database with sample data
@st.cache_resource
def get_database():
    conn = sqlite3.connect(':memory:')
    
    # Companies table
    companies = pd.DataFrame({
        "Ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"],
        "Company": ["Apple", "Google", "Microsoft", "Tesla", "Amazon", "Meta", "Nvidia"],
        "Sector": ["Tech", "Tech", "Tech", "Auto", "Retail", "Social", "Tech"],
        "Revenue_B": [394, 283, 211, 97, 514, 117, 44]
    })
    companies.to_sql("companies", conn, index=False)
    
    # Earnings table
    earnings = pd.DataFrame({
        "Ticker": ["AAPL","AAPL","AAPL","AAPL","MSFT","MSFT","TSLA","TSLA"],
        "Quarter": ["Q1-2023","Q2-2023","Q3-2023","Q4-2023","Q1-2023","Q2-2023","Q1-2023","Q2-2023"],
        "EPS": [1.52, 1.26, 1.46, 2.18, 2.45, 2.69, 0.85, 0.91],
        "Revenue_B": [117.2, 94.8, 89.5, 119.6, 52.7, 56.2, 23.3, 24.9]
    })
    earnings.to_sql("earnings", conn, index=False)
    
    return conn

conn = get_database()
st.success("✅ Sample database loaded!")

# Sidebar
with st.sidebar:
    st.title("📁 Database Info")
    st.write("**Tables:** companies, earnings")
    st.write("**Sample data loaded**")
    
    # API Key
    st.divider()
    st.title("🔑 API Status")
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            client = Groq(api_key=api_key)
            st.success("✅ Groq API Connected")
        else:
            st.error("❌ Groq API Key missing")
    except:
        st.error("❌ Groq API Key missing")

# Show data preview
with st.expander("🔍 Preview Data"):
    tab1, tab2 = st.tabs(["Companies", "Earnings"])
    with tab1:
        st.dataframe(pd.read_sql("SELECT * FROM companies", conn))
    with tab2:
        st.dataframe(pd.read_sql("SELECT * FROM earnings", conn))

# Simple query function
def run_query(sql):
    try:
        df = pd.read_sql(sql, conn)
        return df, None
    except Exception as e:
        return None, str(e)

# Query input
st.divider()
st.subheader("💬 Ask a Question")

question = st.text_input("Enter your question:", placeholder="e.g., Which company has the highest revenue?")

if st.button("Run Query") and question:
    # Simple hardcoded responses for demo
    if "highest revenue" in question.lower():
        df = pd.read_sql("SELECT Company, Revenue_B FROM companies ORDER BY Revenue_B DESC LIMIT 1", conn)
        st.write("✅ Result:")
        st.dataframe(df)
    elif "tech sector" in question.lower():
        df = pd.read_sql("SELECT COUNT(*) as count FROM companies WHERE Sector = 'Tech'", conn)
        st.write("✅ Result:")
        st.dataframe(df)
    elif "tesla" in question.lower() and "eps" in question.lower():
        df = pd.read_sql("SELECT EPS FROM earnings WHERE Ticker = 'TSLA' AND Quarter = 'Q1-2023'", conn)
        st.write("✅ Result:")
        st.dataframe(df)
    else:
        st.info("For full NLP capabilities, please add your Groq API key")
