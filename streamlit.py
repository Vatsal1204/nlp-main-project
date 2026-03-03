import os
import sqlite3
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# Database path
DB_PATH = "data/finance.db"

# Function to build database directly (no imports needed)
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
                                progress=False, auto_adjust=True)
                
                if not df.empty:
                    # Flatten columns
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df["Ticker"] = ticker
                    df.reset_index(inplace=True)
                    all_data.append(df)
                
                # Update progress
                progress_bar.progress((i + 1) / len(tickers))
            except Exception as e:
                st.warning(f"⚠️ Could not download {ticker}: {e}")
        
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
