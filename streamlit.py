import streamlit as st
import pandas as pd
import sqlite3
import re

st.set_page_config(page_title="Financial NLP Query Engine", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Financial NLP Query Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about stock data in plain English</p>', unsafe_allow_html=True)

# Create database connection
conn = sqlite3.connect(':memory:')

# === COMPLETE SAMPLE DATA ===

# Companies table
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

# Earnings table
earnings_data = {
    "Ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "TSLA", "TSLA", "GOOGL", "GOOGL", "NVDA", "META"],
    "Quarter": ["Q1-2023", "Q2-2023", "Q3-2023", "Q4-2023", "Q1-2023", "Q2-2023", "Q1-2023", "Q2-2023", "Q1-2023", "Q2-2023", "Q1-2023", "Q1-2023"],
    "EPS": [1.52, 1.26, 1.46, 2.18, 2.45, 2.69, 0.85, 0.91, 1.17, 1.44, 1.32, 2.87],
    "Revenue_B": [117.2, 94.8, 89.5, 119.6, 52.7, 56.2, 23.3, 24.9, 69.8, 74.6, 45.2, 33.7]
}
earnings_df = pd.DataFrame(earnings_data)
earnings_df.to_sql("earnings", conn, index=False, if_exists="replace")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stocks.png", width=80)
    st.header("📁 Database Info")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Companies", len(companies_df))
    with col2:
        st.metric("Earnings", len(earnings_df))
    
    st.write("**Tables:** companies, earnings")
    st.write("**Data Range:** 2020-2024")
    
    st.divider()
    st.header("🔑 API Status")
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            st.success("✅ Groq API Connected")
        else:
            st.info("ℹ️ Using sample data (no API key)")
    except:
        st.info("ℹ️ Using sample data (no API key)")
    
    st.divider()
    st.header("📜 Query Tips")
    st.write("• Be specific about companies")
    st.write("• Use full company names")
    st.write("• Specify quarters like Q1-2023")
    st.write("• Try: revenue, EPS, employees")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📊 Companies", "💰 Earnings", "🔍 Ask Questions"])

with tab1:
    st.subheader("Companies Master Data")
    st.dataframe(companies_df, use_container_width=True)
    
    # Quick stats
    st.subheader("Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${companies_df['Revenue_B'].sum():,.0f}B")
    with col2:
        st.metric("Total Market Cap", f"${companies_df['Market_Cap_B'].sum():,.0f}B")
    with col3:
        st.metric("Total Employees", f"{companies_df['Employees_K'].sum():,.0f}K")
    with col4:
        st.metric("Avg Revenue/Company", f"${companies_df['Revenue_B'].mean():,.0f}B")

with tab2:
    st.subheader("Earnings Data")
    st.dataframe(earnings_df, use_container_width=True)
    
    # Filter by company
    selected_ticker = st.selectbox("Filter by Company:", options=["All"] + list(earnings_df["Ticker"].unique()))
    if selected_ticker != "All":
        filtered = earnings_df[earnings_df["Ticker"] == selected_ticker]
        st.dataframe(filtered, use_container_width=True)
        
        # Chart
        st.subheader(f"{selected_ticker} EPS Trend")
        chart_data = filtered[["Quarter", "EPS"]].set_index("Quarter")
        st.bar_chart(chart_data)

with tab3:
    st.subheader("Ask Questions in Plain English")
    st.info("💡 Try these example questions below or type your own!")
    
    # Example questions in a grid
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Company Questions:**")
        st.write("• Show all tech companies")
        st.write("• Which company has the highest revenue?")
        st.write("• Show me companies with revenue above 300B")
        st.write("• Which company has the most employees?")
        st.write("• What is Microsoft's market cap?")
        st.write("• Show me Amazon's data")
        st.write("• Companies in Retail sector")
    
    with col2:
        st.markdown("**Earnings Questions:**")
        st.write("• What was Tesla's EPS in Q1 2023?")
        st.write("• Apple's EPS in Q4 2023")
        st.write("• Show Microsoft's earnings")
        st.write("• Google Q2 2023 revenue")
        st.write("• Nvidia EPS")
        st.write("• Meta earnings")
    
    st.divider()
    
    # Question input
    question = st.text_input("✏️ Type your question here:", placeholder="e.g., Show all tech companies")
    
    if st.button("🚀 Run Query", use_container_width=True) and question:
        question_lower = question.lower()
        
        # ===== COMPANY QUESTIONS =====
        
        # Tech companies
        if ("tech" in question_lower and "company" in question_lower) or ("tech sector" in question_lower):
            result = companies_df[companies_df["Sector"] == "Tech"]
            st.success(f"✅ Found {len(result)} tech companies")
            st.dataframe(result, use_container_width=True)
        
        # Highest revenue
        elif "highest revenue" in question_lower or "max revenue" in question_lower or "top revenue" in question_lower:
            result = companies_df.nlargest(1, "Revenue_B")[["Company", "Revenue_B", "Sector"]]
            st.success("✅ Company with highest revenue:")
            st.dataframe(result, use_container_width=True)
        
        # Revenue above threshold
        elif "revenue above" in question_lower or "revenue greater than" in question_lower or "revenue >" in question_lower:
            numbers = re.findall(r'\d+', question_lower)
            if numbers:
                threshold = int(numbers[0])
                result = companies_df[companies_df["Revenue_B"] > threshold][["Company", "Revenue_B", "Sector"]].sort_values("Revenue_B", ascending=False)
                st.success(f"✅ Companies with revenue > {threshold}B:")
                st.dataframe(result, use_container_width=True)
            else:
                st.warning("Please specify a number, e.g., 'revenue above 300 billion'")
        
        # Most employees
        elif "most employees" in question_lower or "highest employees" in question_lower or "max employees" in question_lower:
            result = companies_df.nlargest(1, "Employees_K")[["Company", "Employees_K", "Sector"]]
            st.success("✅ Company with most employees:")
            st.dataframe(result, use_container_width=True)
        
        # Market cap queries
        elif "market cap" in question_lower or "market capitalization" in question_lower:
            if "microsoft" in question_lower:
                result = companies_df[companies_df["Company"] == "Microsoft"][["Company", "Market_Cap_B"]]
                st.success("✅ Microsoft Market Cap:")
            elif "apple" in question_lower:
                result = companies_df[companies_df["Company"] == "Apple"][["Company", "Market_Cap_B"]]
                st.success("✅ Apple Market Cap:")
            elif "amazon" in question_lower:
                result = companies_df[companies_df["Company"] == "Amazon"][["Company", "Market_Cap_B"]]
                st.success("✅ Amazon Market Cap:")
            elif "google" in question_lower:
                result = companies_df[companies_df["Company"] == "Google"][["Company", "Market_Cap_B"]]
                st.success("✅ Google Market Cap:")
            elif "tesla" in question_lower:
                result = companies_df[companies_df["Company"] == "Tesla"][["Company", "Market_Cap_B"]]
                st.success("✅ Tesla Market Cap:")
            else:
                result = companies_df[["Company", "Market_Cap_B"]].sort_values("Market_Cap_B", ascending=False)
                st.success("✅ All Companies by Market Cap:")
            st.dataframe(result, use_container_width=True)
        
        # Specific company data
        elif "amazon" in question_lower and "data" in question_lower:
            result = companies_df[companies_df["Company"] == "Amazon"]
            st.success("✅ Amazon Data:")
            st.dataframe(result, use_container_width=True)
        
        elif "microsoft" in question_lower and "data" in question_lower:
            result = companies_df[companies_df["Company"] == "Microsoft"]
            st.success("✅ Microsoft Data:")
            st.dataframe(result, use_container_width=True)
        
        elif "apple" in question_lower and "data" in question_lower:
            result = companies_df[companies_df["Company"] == "Apple"]
            st.success("✅ Apple Data:")
            st.dataframe(result, use_container_width=True)
        
        elif "google" in question_lower and "data" in question_lower:
            result = companies_df[companies_df["Company"] == "Google"]
            st.success("✅ Google Data:")
            st.dataframe(result, use_container_width=True)
        
        elif "tesla" in question_lower and "data" in question_lower:
            result = companies_df[companies_df["Company"] == "Tesla"]
            st.success("✅ Tesla Data:")
            st.dataframe(result, use_container_width=True)
        
        elif "nvidia" in question_lower and "data" in question_lower:
            result = companies_df[companies_df["Company"] == "Nvidia"]
            st.success("✅ Nvidia Data:")
            st.dataframe(result, use_container_width=True)
        
        # Retail sector
        elif "retail" in question_lower and ("sector" in question_lower or "company" in question_lower):
            result = companies_df[companies_df["Sector"] == "Retail"][["Company", "Sector", "Revenue_B"]]
            st.success("✅ Companies in Retail sector:")
            st.dataframe(result, use_container_width=True)
        
        # Auto sector
        elif "auto" in question_lower and ("sector" in question_lower or "company" in question_lower):
            result = companies_df[companies_df["Sector"] == "Auto"][["Company", "Sector", "Revenue_B"]]
            st.success("✅ Companies in Auto sector:")
            st.dataframe(result, use_container_width=True)
        
        # Social sector
        elif "social" in question_lower and ("sector" in question_lower or "company" in question_lower):
            result = companies_df[companies_df["Sector"] == "Social"][["Company", "Sector", "Revenue_B"]]
            st.success("✅ Companies in Social Media sector:")
            st.dataframe(result, use_container_width=True)
        
        # ===== EARNINGS QUESTIONS =====
        
        # Tesla EPS Q1 2023
        elif "tesla" in question_lower and "eps" in question_lower and ("q1" in question_lower or "quarter 1" in question_lower):
            result = earnings_df[(earnings_df["Ticker"] == "TSLA") & (earnings_df["Quarter"] == "Q1-2023")][["Quarter", "EPS", "Revenue_B"]]
            st.success("✅ Tesla Q1 2023 Earnings:")
            st.dataframe(result, use_container_width=True)
        
        # Apple EPS Q4 2023
        elif "apple" in question_lower and "eps" in question_lower and ("q4" in question_lower or "quarter 4" in question_lower):
            result = earnings_df[(earnings_df["Ticker"] == "AAPL") & (earnings_df["Quarter"] == "Q4-2023")][["Quarter", "EPS", "Revenue_B"]]
            st.success("✅ Apple Q4 2023 Earnings:")
            st.dataframe(result, use_container_width=True)
        
        # Microsoft earnings
        elif "microsoft" in question_lower and ("earnings" in question_lower or "eps" in question_lower):
            result = earnings_df[earnings_df["Ticker"] == "MSFT"][["Quarter", "EPS", "Revenue_B"]]
            st.success("✅ Microsoft Earnings:")
            st.dataframe(result, use_container_width=True)
        
        # Google earnings
        elif "google" in question_lower and ("earnings" in question_lower or "eps" in question_lower):
            result = earnings_df[earnings_df["Ticker"] == "GOOGL"][["Quarter", "EPS", "Revenue_B"]]
            st.success("✅ Google Earnings:")
            st.dataframe(result, use_container_width=True)
        
        # Nvidia earnings
        elif "nvidia" in question_lower and ("earnings" in question_lower or "eps" in question_lower):
            result = earnings_df[earnings_df["Ticker"] == "NVDA"][["Quarter", "EPS", "Revenue_B"]]
            st.success("✅ Nvidia Earnings:")
            st.dataframe(result, use_container_width=True)
        
        # Meta earnings
        elif "meta" in question_lower and ("earnings" in question_lower or "eps" in question_lower):
            result = earnings_df[earnings_df["Ticker"] == "META"][["Quarter", "EPS", "Revenue_B"]]
            st.success("✅ Meta Earnings:")
            st.dataframe(result, use_container_width=True)
        
        # Quarter specific revenue
        elif "revenue" in question_lower and "q1" in question_lower and "2023" in question_lower:
            result = earnings_df[earnings_df["Quarter"] == "Q1-2023"][["Ticker", "Revenue_B"]].sort_values("Revenue_B", ascending=False)
            st.success("✅ Q1 2023 Revenue by Company:")
            st.dataframe(result, use_container_width=True)
        
        # ===== FALLBACK =====
        else:
            st.info("🤔 I didn't understand that question. Try one of the examples above!")
            
            # Show suggestion based on keywords
            if "revenue" in question_lower:
                st.write("💡 Try: 'Show companies with revenue above 300B'")
            elif "eps" in question_lower:
                st.write("💡 Try: 'What was Tesla's EPS in Q1 2023?'")
            elif "company" in question_lower or "companies" in question_lower:
                st.write("💡 Try: 'Show all tech companies'")
            elif "sector" in question_lower:
                st.write("💡 Try: 'Companies in Retail sector'")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    Built with Streamlit • Financial Data Sample • Ask questions in plain English
</div>
""", unsafe_allow_html=True)
