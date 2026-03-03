import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="Financial NLP", page_icon="📊", layout="wide")

st.title("📊 Financial NLP Query Engine")
st.write("Ask questions about stock data in plain English")

# Create database connection
conn = sqlite3.connect(':memory:')

# === COMPLETE SAMPLE DATA ===

# 1. Companies table
companies_data = {
    "Ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"],
    "Company": ["Apple", "Google", "Microsoft", "Tesla", "Amazon", "Meta", "Nvidia"],
    "Sector": ["Tech", "Tech", "Tech", "Auto", "Retail", "Social", "Tech"],
    "Revenue_B": [394, 283, 211, 97, 514, 117, 44],
    "Market_Cap_B": [2800, 1700, 2500, 800, 1600, 900, 1200],
    "Employees_K": [161, 182, 221, 127, 1541, 86, 26]
}
companies_df = pd.DataFrame(companies_data)
companies_df.to_sql("companies", conn, index=False, if_exists="replace")

# 2. Earnings table
earnings_data = {
    "Ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "TSLA", "TSLA"],
    "Quarter": ["Q1-2023", "Q2-2023", "Q3-2023", "Q4-2023", "Q1-2023", "Q2-2023", "Q1-2023", "Q2-2023"],
    "EPS": [1.52, 1.26, 1.46, 2.18, 2.45, 2.69, 0.85, 0.91],
    "Revenue_B": [117.2, 94.8, 89.5, 119.6, 52.7, 56.2, 23.3, 24.9]
}
earnings_df = pd.DataFrame(earnings_data)
earnings_df.to_sql("earnings", conn, index=False, if_exists="replace")

# Sidebar
with st.sidebar:
    st.header("📁 Database Info")
    st.write(f"**Companies:** {len(companies_df)} rows")
    st.write(f"**Earnings:** {len(earnings_df)} rows")
    st.write("**Tables:** companies, earnings")
    
    st.divider()
    st.header("🔑 API Status")
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            st.success("✅ API Key found")
        else:
            st.info("ℹ️ No API key (using sample data)")
    except:
        st.info("ℹ️ No API key (using sample data)")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Companies", "💰 Earnings", "🔍 Query"])

with tab1:
    st.subheader("Companies Data")
    st.dataframe(companies_df, use_container_width=True)
    
    # Simple filters
    col1, col2 = st.columns(2)
    with col1:
        sectors = st.multiselect("Filter by Sector", options=companies_df["Sector"].unique())
        if sectors:
            filtered = companies_df[companies_df["Sector"].isin(sectors)]
            st.dataframe(filtered, use_container_width=True)
    with col2:
        min_revenue = st.slider("Min Revenue (B$)", 0, 600, 0)
        filtered2 = companies_df[companies_df["Revenue_B"] >= min_revenue]
        st.dataframe(filtered2, use_container_width=True)

with tab2:
    st.subheader("Earnings Data")
    st.dataframe(earnings_df, use_container_width=True)
    
    # Filter by ticker
    ticker = st.selectbox("Select Ticker", options=earnings_df["Ticker"].unique())
    if ticker:
        filtered = earnings_df[earnings_df["Ticker"] == ticker]
        st.dataframe(filtered, use_container_width=True)

with tab3:
    st.subheader("Ask a Question")
    st.write("Example questions:")
    st.write("- Show all tech companies")
    st.write("- Which company has the highest revenue?")
    st.write("- What was Tesla's EPS in Q1 2023?")
    
    question = st.text_input("Your question:")
    
    if st.button("Run Query") and question:
        question_lower = question.lower()
        
        if "tech" in question_lower and "company" in question_lower:
            result = companies_df[companies_df["Sector"] == "Tech"]
            st.success(f"Found {len(result)} tech companies")
            st.dataframe(result)
            
        elif "highest revenue" in question_lower:
            result = companies_df.nlargest(1, "Revenue_B")
            st.success("Company with highest revenue:")
            st.dataframe(result)
            
        elif "tesla" in question_lower and "eps" in question_lower and "q1 2023" in question_lower:
            result = earnings_df[(earnings_df["Ticker"] == "TSLA") & (earnings_df["Quarter"] == "Q1-2023")]
            st.success("Tesla EPS Q1 2023:")
            st.dataframe(result)
            
        elif "total" in question_lower and "revenue" in question_lower:
            total = companies_df["Revenue_B"].sum()
            st.metric("Total Revenue (B$)", f"{total:,.0f}")
            
        else:
            st.info("Try one of the example questions above!")
