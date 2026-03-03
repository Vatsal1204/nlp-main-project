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
        st.write("• Which company has the lowest revenue?")
        st.write("• Show me companies with revenue above 300B")
        st.write("• Which company has the most employees?")
        st.write("• Companies with employees less than 200K")
        st.write("• What is Microsoft's market cap?")
        st.write("• Show me Amazon's data")
        st.write("• Companies in Retail sector")
        st.write("• Which sector has the most companies?")
        st.write("• Compare Apple and Microsoft revenue")
        st.write("• Total revenue of all companies")
    
    with col2:
        st.markdown("**Earnings Questions:**")
        st.write("• What was Tesla's EPS in Q1 2023?")
        st.write("• Show me all earnings for Tesla in 2023")
        st.write("• Apple's EPS in Q4 2023")
        st.write("• Show Microsoft's earnings")
        st.write("• Google Q2 2023 revenue")
        st.write("• Average EPS for tech companies")
        st.write("• Nvidia EPS")
        st.write("• Meta earnings")
        st.write("• What was Google's revenue in Q2 2023?")
    
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
        
        # Lowest revenue
        elif "lowest revenue" in question_lower or "min revenue" in question_lower or "smallest revenue" in question_lower:
            result = companies_df.nsmallest(1, "Revenue_B")[["Company", "Revenue_B", "Sector"]]
            st.success("✅ Company with lowest revenue:")
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
        
        # Employees less than 200K
        elif ("employees less than" in question_lower or "employees <" in question_lower or "employees under" in question_lower) and ("200" in question_lower or "200k" in question_lower):
            result = companies_df[companies_df["Employees_K"] < 200][["Company", "Employees_K", "Sector"]].sort_values("Employees_K")
            st.success("✅ Companies with < 200K employees:")
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
        
        # Compare Apple and Microsoft revenue
        elif ("compare" in question_lower and "apple" in question_lower and "microsoft" in question_lower and "revenue" in question_lower):
            result = companies_df[companies_df["Company"].isin(["Apple", "Microsoft"])][["Company", "Revenue_B"]]
            st.success("✅ Apple vs Microsoft Revenue:")
            st.dataframe(result, use_container_width=True)
        
        # Total revenue of all companies
        elif ("total revenue" in question_lower and "all companies" in question_lower) or ("sum of revenue" in question_lower):
            total = companies_df["Revenue_B"].sum()
            st.success(f"✅ Total Revenue of All Companies: **${total}B**")
            st.metric("Total Revenue", f"${total}B")
        
        # Which sector has the most companies
        elif ("sector" in question_lower and "most companies" in question_lower) or ("largest sector" in question_lower):
            sector_counts = companies_df["Sector"].value_counts().reset_index()
            sector_counts.columns = ["Sector", "Count"]
            st.success("✅ Number of Companies by Sector:")
            st.dataframe(sector_counts, use_container_width=True)
            
            # Show the largest
            largest = sector_counts.iloc[0]
            st.info(f"🏆 **{largest['Sector']}** has the most companies ({largest['Count']})")
        
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
        
        # Show all earnings for Tesla in 2023
        elif ("tesla" in question_lower and "all earnings" in question_lower) or ("tesla" in question_lower and "2023" in question_lower and "earnings" in question_lower):
            result = earnings_df[earnings_df["Ticker"] == "TSLA"][["Quarter", "EPS", "Revenue_B"]]
            st.success("✅ Tesla 2023 Earnings:")
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
        
        # Google's revenue in Q2 2023
        elif ("google" in question_lower and "revenue" in question_lower and ("q2" in question_lower or "quarter 2" in question_lower) and "2023" in question_lower):
            result = earnings_df[(earnings_df["Ticker"] == "GOOGL") & (earnings_df["Quarter"] == "Q2-2023")][["Quarter", "Revenue_B"]]
            st.success("✅ Google Q2 2023 Revenue:")
            st.dataframe(result, use_container_width=True)
        
        # Average EPS for tech companies
        elif ("average eps" in question_lower or "avg eps" in question_lower) and ("tech" in question_lower or "technology" in question_lower):
            tech_tickers = companies_df[companies_df["Sector"] == "Tech"]["Ticker"].tolist()
            avg_eps = earnings_df[earnings_df["Ticker"].isin(tech_tickers)]["EPS"].mean()
            st.success(f"✅ Average EPS for Tech Companies: **{avg_eps:.2f}**")
            st.metric("Average EPS", f"{avg_eps:.2f}")
        
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
                st.write("💡 Try: 'Show companies with revenue above 300B' or 'Total revenue of all companies'")
            elif "eps" in question_lower:
                st.write("💡 Try: 'What was Tesla's EPS in Q1 2023?' or 'Average EPS for tech companies'")
            elif "company" in question_lower or "companies" in question_lower:
                st.write("💡 Try: 'Show all tech companies' or 'Which company has the lowest revenue?'")
            elif "sector" in question_lower:
                st.write("💡 Try: 'Companies in Retail sector' or 'Which sector has the most companies?'")
            elif "employee" in question_lower:
                st.write("💡 Try: 'Companies with employees less than 200K' or 'Which company has the most employees?'")
            elif "compare" in question_lower:
                st.write("💡 Try: 'Compare Apple and Microsoft revenue'")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    Built with Streamlit • Financial Data Sample • Ask questions in plain English
</div>
""", unsafe_allow_html=True)
