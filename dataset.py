import yfinance as yf
import pandas as pd
import sqlite3
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

DB_PATH = "data/finance.db"

def download_stock_data():
    print("📥 Downloading stock data...")
    
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
    
    all_data = []
    for ticker in tickers:
        df = yf.download(ticker, start="2020-01-01", end="2024-12-31", 
                        progress=False, auto_adjust=True)
        # Flatten multi-level columns
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        # Keep only needed columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["Ticker"] = ticker
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        all_data.append(df)
    
    stocks_df = pd.concat(all_data, ignore_index=True)
    # Final check - drop any duplicate columns
    stocks_df = stocks_df.loc[:, ~stocks_df.columns.duplicated()]
    print(f"✅ Downloaded {len(stocks_df)} stock records")
    return stocks_df

def create_company_data():
    print("🏢 Creating company data...")
    
    companies = {
        "Ticker":       ["AAPL",  "GOOGL", "MSFT",  "TSLA",  "AMZN",  "META",  "NVDA"],
        "Company":      ["Apple", "Google","Microsoft","Tesla","Amazon","Meta","Nvidia"],
        "Sector":       ["Tech",  "Tech",  "Tech",  "Auto",  "Retail","Social","Tech"],
        "Market_Cap_B": [2800,    1700,    2500,    800,     1600,    900,     1200],
        "Revenue_B":    [394,     283,     211,     97,      514,     117,     44],
        "Employees_K":  [161,     182,     221,     127,     1541,    86,      26]
    }
    return pd.DataFrame(companies)

def create_earnings_data():
    print("📊 Creating earnings data...")
    
    earnings = {
        "Ticker":    ["AAPL","AAPL","AAPL","AAPL","MSFT","MSFT","MSFT","MSFT",
                      "GOOGL","GOOGL","GOOGL","GOOGL","TSLA","TSLA","TSLA","TSLA"],
        "Quarter":   ["Q1-2023","Q2-2023","Q3-2023","Q4-2023"] * 4,
        "EPS":       [1.52, 1.26, 1.46, 2.18, 2.45, 2.69, 2.99, 2.93,
                      1.17, 1.44, 1.55, 1.64, 0.85, 0.91, 0.66, 2.27],
        "Revenue_B": [117.2, 94.8, 89.5, 119.6, 52.7, 56.2, 56.5, 62.0,
                      69.8, 74.6, 76.7, 86.3, 23.3, 24.9, 25.2, 25.2],
        "Net_Income_B":[24.2, 19.9, 22.9, 33.9, 18.3, 20.1, 22.3, 21.9,
                        15.1, 18.4, 19.7, 20.4, 2.5, 2.7, 1.9, 7.9]
    }
    return pd.DataFrame(earnings)

def build_database():
    print("\n🔨 Building Financial Database...")
    print("=" * 40)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Table 1: Stock prices
    stocks_df = download_stock_data()
    stocks_df.to_sql("stocks", conn, if_exists="replace", index=False)
    print(f"✅ 'stocks' table created — {len(stocks_df)} rows")
    
    # Table 2: Company info
    companies_df = create_company_data()
    companies_df.to_sql("companies", conn, if_exists="replace", index=False)
    print(f"✅ 'companies' table created — {len(companies_df)} rows")
    
    # Table 3: Earnings
    earnings_df = create_earnings_data()
    earnings_df.to_sql("earnings", conn, if_exists="replace", index=False)
    print(f"✅ 'earnings' table created — {len(earnings_df)} rows")
    
    conn.close()
    print("\n🎉 Database built successfully at:", DB_PATH)

def verify_database():
    print("\n🔍 Verifying Database...")
    print("=" * 40)
    
    conn = sqlite3.connect(DB_PATH)
    
    tables = ["stocks", "companies", "earnings"]
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
        print(f"\n📋 Table: {table}")
        print(df.to_string())
    
    conn.close()

if __name__ == "__main__":
    build_database()
    verify_database()