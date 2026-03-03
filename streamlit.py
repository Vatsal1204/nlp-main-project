import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="Financial NLP", page_icon="📊")

st.title("📊 Financial NLP Query Engine")
st.write("Ask questions about stock data in plain English")

# Create database AND data in one simple step
conn = sqlite3.connect(':memory:')

# Create companies table
companies_data = {
    "Ticker": ["AAPL", "GOOGL", "MSFT", "TSLA"],
    "Company": ["Apple", "Google", "Microsoft", "Tesla"],
    "Sector": ["Tech", "Tech", "Tech", "Auto"],
    "Revenue_B": [394, 283, 211, 97]
}
companies_df = pd.DataFrame(companies_data)
companies_df.to_sql("companies", conn, index=False, if_exists="replace")

# Show the data directly (not through SQL)
st.subheader("Companies Data")
st.dataframe(companies_df)

# Simple query example
st.subheader("Query Example")
if st.button("Show Tech Companies"):
    tech_df = companies_df[companies_df["Sector"] == "Tech"]
    st.dataframe(tech_df)

# API Key status (optional)
with st.sidebar:
    st.write("API Status")
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            st.success("✅ API Key found")
        else:
            st.info("ℹ️ No API key (using sample data)")
    except:
        st.info("ℹ️ No API key (using sample data)")
