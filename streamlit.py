import sys
import traceback

# Import everything at the top - ONCE
try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import yfinance as yf
    from groq import Groq
    from dotenv import load_dotenv
    import os
    import sqlite3
    from datetime import datetime
except Exception as e:
    print("IMPORT ERROR:", e)
    print(traceback.format_exc())
    raise e

# Load environment variables
load_dotenv()

# Database path
DB_PATH = "data/finance.db"

# Page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Financial NLP Query Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# Function to build database directly
def build_database():
    """Build database directly in the app"""
    
    # Create data folder
    os.makedirs("data", exist_ok=True)
    
    # Create connection
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Download stock data with progress
        st.info("📥 Downloading stock data (this takes 30-60 seconds)...")
        tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
        all_data = []
        
        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            try:
                df = yf.download(ticker, start="2020-01-01", end="2024-12-31", 
                                progress=False, auto_adjust=True, timeout=10)
                
                if not df.empty:
                    # Flatten columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df["Ticker"] = ticker
                    df.reset_index(inplace=True)
                    all_data.append(df)
                
                # Update progress
                progress_bar.progress((i + 1) / len(tickers))
            except Exception as e:
                st.warning(f"⚠️ Could not download {ticker}: {e}")
                progress_bar.progress((i + 1) / len(tickers))
        
        if all_data:
            stocks_df = pd.concat(all_data, ignore_index=True)
            stocks_df = stocks_df.loc[:, ~stocks_df.columns.duplicated()]
            stocks_df.to_sql("stocks", conn, if_exists="replace", index=False)
            st.success(f"✅ Stocks table created with {len(stocks_df)} records")
        
        # Create companies table
        companies = pd.DataFrame({
            "Ticker": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"],
            "Company": ["Apple", "Google", "Microsoft", "Tesla", "Amazon", "Meta", "Nvidia"],
            "Sector": ["Tech", "Tech", "Tech", "Auto", "Retail", "Social", "Tech"],
            "Market_Cap_B": [2800, 1700, 2500, 800, 1600, 900, 1200],
            "Revenue_B": [394, 283, 211, 97, 514, 117, 44],
            "Employees_K": [161, 182, 221, 127, 1541, 86, 26]
        })
        companies.to_sql("companies", conn, if_exists="replace", index=False)
        st.success(f"✅ Companies table created with {len(companies)} records")
        
        # Create earnings table
        earnings = pd.DataFrame({
            "Ticker": ["AAPL","AAPL","AAPL","AAPL","MSFT","MSFT","MSFT","MSFT",
                      "GOOGL","GOOGL","GOOGL","GOOGL","TSLA","TSLA","TSLA","TSLA"],
            "Quarter": ["Q1-2023","Q2-2023","Q3-2023","Q4-2023"] * 4,
            "EPS": [1.52, 1.26, 1.46, 2.18, 2.45, 2.69, 2.99, 2.93,
                    1.17, 1.44, 1.55, 1.64, 0.85, 0.91, 0.66, 2.27],
            "Revenue_B": [117.2, 94.8, 89.5, 119.6, 52.7, 56.2, 56.5, 62.0,
                          69.8, 74.6, 76.7, 86.3, 23.3, 24.9, 25.2, 25.2],
            "Net_Income_B": [24.2, 19.9, 22.9, 33.9, 18.3, 20.1, 22.3, 21.9,
                             15.1, 18.4, 19.7, 20.4, 2.5, 2.7, 1.9, 7.9]
        })
        earnings.to_sql("earnings", conn, if_exists="replace", index=False)
        st.success(f"✅ Earnings table created with {len(earnings)} records")
        
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"❌ Database build error: {e}")
        conn.close()
        return False

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Financial NLP Query Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about stock data in plain English</p>', unsafe_allow_html=True)
    
    # Check and build database
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Database not found. Building it now...")
        with st.spinner("Building database (this will take 1-2 minutes)..."):
            success = build_database()
            if success:
                st.success("✅ Database built successfully! Refreshing...")
                st.rerun()
            else:
                st.error("❌ Failed to build database. Please check the error above.")
                st.stop()
    
    # Rest of your app UI here
    st.success("✅ Database is ready! Add your query interface here.")

# Run the app
if __name__ == "__main__":
    main()
