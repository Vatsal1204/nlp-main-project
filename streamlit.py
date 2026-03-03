import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="Financial NLP", page_icon="📊")

st.title("📊 Financial NLP Query Engine")
st.write("Ask questions about stock data in plain English")

# Use a simple in-memory database with sample data
conn = sqlite3.connect(':memory:')

# Create sample data directly
companies = pd.DataFrame({
    "Ticker": ["AAPL", "GOOGL", "MSFT", "TSLA"],
    "Company": ["Apple", "Google", "Microsoft", "Tesla"],
    "Sector": ["Tech", "Tech", "Tech", "Auto"],
    "Revenue_B": [394, 283, 211, 97]
})
companies.to_sql("companies", conn, index=False)

earnings = pd.DataFrame({
    "Ticker": ["AAPL", "AAPL", "TSLA", "TSLA"],
    "Quarter": ["Q1-2023", "Q2-2023", "Q1-2023", "Q2-2023"],
    "EPS": [1.52, 1.26, 0.85, 0.91]
})
earnings.to_sql("earnings", conn, index=False)

st.success("✅ Sample database loaded!")
st.dataframe(companies)
