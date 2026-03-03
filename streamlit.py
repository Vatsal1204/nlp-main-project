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
    "Ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "TSLA", "TSLA", "GOOGL", "GOOGL"],
    "Quarter": ["Q1-2023", "Q2-2023", "Q3-2023", "Q4-2023", "Q1-2023", "Q2-2023", "Q1-2023", "Q2-2023", "Q1-2023", "Q2-2023"],
    "EPS": [1.52, 1.26, 1.46, 2.18, 2.45, 2.69, 0.85, 0.91, 1.17, 1.44],
    "Revenue_B": [117.2, 94.8, 89.5, 119.6, 52.7, 56.2, 23.3, 24.9, 69.8, 74.6]
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

with tab2:
    st.subheader("Earnings Data")
    st.dataframe(earnings_df, use_container_width=True)

with tab3:
    st.subheader("Ask a Question")
    st.write("**Try these example questions:**")
    st.write("1. Show all tech companies")
    st.write("2. Which company has the highest revenue?")
    st.write("3. What was Tesla's EPS in Q1 2023?")
    st.write("4. Show me Amazon's data")
    st.write("5. What is Microsoft's market cap?")
    
    question = st.text_input("Your question:")
    
    if st.button("Run Query") and question:
        question_lower = question.lower()
        
        # Question 1: Show all tech companies
        if "tech" in question_lower and "company" in question_lower:
            result = companies_df[companies_df["Sector"] == "Tech"]
            st.success(f"✅ Found {len(result)} tech companies")
            st.dataframe(result)
        
        # Question 2: Which company has the highest revenue?
        elif "highest revenue" in question_lower:
            result = companies_df.nlargest(1, "Revenue_B")[["Company", "Revenue_B"]]
            st.success("✅ Company with highest revenue:")
            st.dataframe(result)
        
        # Question 3: Tesla's EPS in Q1 2023
        elif "tesla" in question_lower and "eps" in question_lower and "q1" in question_lower:
            result = earnings_df[(earnings_df["Ticker"] == "TSLA") & (earnings_df["Quarter"] == "Q1-2023")][["EPS", "Revenue_B"]]
            st.success("✅ Tesla Q1 2023 EPS:")
            st.dataframe(result)
        
        # Show specific company data
        elif "amazon" in question_lower:
            result = companies_df[companies_df["Company"] == "Amazon"]
            st.success("✅ Amazon data:")
            st.dataframe(result)
        
        # Microsoft market cap
        elif "microsoft" in question_lower and "market cap" in question_lower:
            result = companies_df[companies_df["Company"] == "Microsoft"][["Company", "Market_Cap_B"]]
            st.success("✅ Microsoft Market Cap:")
            st.dataframe(result)
        
        # Default response
        else:
            st.info("Try one of the example questions above!")
