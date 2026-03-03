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
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        border-radius: 5px;
    }
    .info-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'client' not in st.session_state:
    st.session_state.client = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# Database functions
@st.cache_data(ttl=3600)
def get_db_schema():
    """Get database schema for context with existence checks"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    schema = ""
    tables = ["stocks", "companies", "earnings"]
    table_info = {}
    
    # First check which tables actually exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        table_info[table] = []
        
        # Check if table exists
        if table not in existing_tables:
            schema += f"Table '{table}': Table does not exist yet\n\n"
            table_info[table].append({"row_count": 0})
            continue
            
        # Get column info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        if not columns:
            schema += f"Table '{table}': No columns found\n\n"
            table_info[table].append({"row_count": 0})
            continue
            
        col_names = [col[1] for col in columns]
        col_types = [col[2] for col in columns]
        
        schema += f"Table '{table}':\n"
        for name, type_ in zip(col_names, col_types):
            schema += f"  - {name} ({type_})\n"
            table_info[table].append({"name": name, "type": type_})
        
        # Get row count (with error handling)
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            schema += f"  → {count} rows\n\n"
            table_info[table].append({"row_count": count})
        except:
            schema += f"  → Unable to count rows\n\n"
            table_info[table].append({"row_count": 0})
    
    conn.close()
    return schema, table_info

@st.cache_data(ttl=300)
def get_table_preview(table_name, limit=10):
    """Get preview of table data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

def run_query(sql):
    """Execute SQL on database"""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        conn.close()
        return None, str(e)

def clean_sql(sql):
    """Clean SQL query returned by LLM"""
    sql = sql.strip()
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.replace("SQLQuery:", "").strip()
    sql = sql.replace("SQL Query:", "").strip()
    return sql

def setup_groq_client():
    """Setup Groq client"""
    try:
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        if not api_key:
            return None
        return Groq(api_key=api_key)
    except:
        return None

def generate_sql(question, client, schema):
    """Generate SQL from natural language"""
    
    few_shot_examples = """
-- Example 1: Complex date filtering (quarters)
Question: Which quarter had the highest revenue for Apple in 2023?
SQL: SELECT Quarter, Revenue_B FROM earnings WHERE Ticker = 'AAPL' AND Quarter LIKE '%2023' ORDER BY Revenue_B DESC LIMIT 1;

-- Example 2: Count with alias
Question: How many companies are in the Tech sector?
SQL: SELECT COUNT(*) as count FROM companies WHERE Sector = 'Tech';

-- Example 3: Year filtering with strftime
Question: What is the average closing price of AAPL in 2023?
SQL: SELECT AVG(Close) as avg_close FROM stocks WHERE Ticker = 'AAPL' AND strftime('%Y', Date) = '2023';

-- Example 4: Join with company name
Question: What was Tesla's EPS in Q1 2023?
SQL: SELECT e.EPS FROM earnings e JOIN companies c ON e.Ticker = c.Ticker WHERE c.Company = 'Tesla' AND e.Quarter = 'Q1-2023';

-- Example 5: Aggregation with grouping
Question: What is the total volume for each stock in 2023?
SQL: SELECT Ticker, SUM(Volume) as total_volume FROM stocks WHERE strftime('%Y', Date) = '2023' GROUP BY Ticker;

-- Example 6: Comparison query
Question: Which stock had the highest closing price in 2024?
SQL: SELECT Ticker, MAX(Close) as max_close FROM stocks WHERE strftime('%Y', Date) = '2024' GROUP BY Ticker ORDER BY max_close DESC LIMIT 1;
"""
    
    prompt = f"""You are a SQLite expert. Generate a SQL query to answer the question.

Database Schema:
{schema}

Here are examples of correct SQL queries for similar questions:
{few_shot_examples}

Important rules:
- Use proper SQLite syntax
- Always use single quotes for strings
- For year filtering: strftime('%Y', Date) = 'YYYY'
- For quarter filtering: Quarter LIKE '%YYYY'
- Always include meaningful column aliases
- Return ONLY the SQL query, no explanations

Question: {question}

SQL Query:"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate only the SQL query, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        sql = clean_sql(response.choices[0].message.content)
        return sql, None
    except Exception as e:
        return None, str(e)

def create_visualization(df, query_type):
    """Create appropriate visualization based on data"""
    if df.empty or len(df.columns) < 1:
        return None
    
    try:
        # Time series data (has Date column)
        if 'Date' in df.columns:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                fig = px.line(df, x='Date', y=numeric_cols[0], 
                             title=f'{numeric_cols[0]} over Time',
                             template='plotly_white')
                return fig
        
        # Bar chart for comparisons
        elif len(df.columns) >= 2 and df.select_dtypes(include=['float64', 'int64']).shape[1] >= 1:
            cat_col = df.select_dtypes(include=['object']).columns[0] if df.select_dtypes(include=['object']).shape[1] > 0 else df.columns[0]
            num_col = df.select_dtypes(include=['float64', 'int64']).columns[0]
            
            fig = px.bar(df, x=cat_col, y=num_col, 
                        title=f'{num_col} by {cat_col}',
                        template='plotly_white')
            return fig
        
        # Pie chart for distributions
        elif len(df.columns) == 2 and 'count' in df.columns[1].lower():
            fig = px.pie(df, values=df.columns[1], names=df.columns[0],
                        title='Distribution',
                        template='plotly_white')
            return fig
            
    except Exception as e:
        return None
    
    return None

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
        st.session_state.db_initialized = True
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
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stocks.png", width=100)
        st.title("📁 Database Info")
        
        # Check if database exists and build if needed
        if not os.path.exists(DB_PATH):
            with st.spinner("Building database for first time..."):
                success = build_database()
                if success:
                    st.success("✅ Database built!")
                    st.rerun()
                else:
                    st.error("❌ Database build failed")
                    st.stop()
        else:
            # Verify database has tables
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()
                
                if not tables:
                    st.warning("⚠️ Database exists but no tables found. Rebuilding...")
                    os.remove(DB_PATH)
                    st.rerun()
                else:
                    st.session_state.db_initialized = True
            except Exception as e:
                st.warning("⚠️ Database corrupt. Rebuilding...")
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)
                st.rerun()
        
        # Only load schema if database is initialized
        if st.session_state.db_initialized:
            with st.spinner("Loading database schema..."):
                schema, table_info = get_db_schema()
            
            # Display table info
            for table, info in table_info.items():
                with st.expander(f"📋 {table.upper()}"):
                    cols = [col for col in info if isinstance(col, dict)]
                    for col in cols[:5]:  # Show first 5 columns
                        st.write(f"• {col['name']} ({col['type']})")
                    row_count = next((item for item in info if 'row_count' in item), None)
                    if row_count and row_count['row_count'] > 0:
                        st.write(f"**Rows:** {row_count['row_count']:,}")
                    else:
                        st.write("**Rows:** 0 (building...)")
        else:
            schema = "Database is being built..."
            table_info = {}
            st.info("⏳ Database is being initialized...")
        
        # API Key status
        st.divider()
        st.title("🔑 API Status")
        client = setup_groq_client()
        if client:
            st.success("✅ Groq API Connected")
            st.session_state.client = client
        else:
            st.error("❌ Groq API Key missing")
            st.info("Add GROQ_API_KEY to .env file or Streamlit secrets")
        
        # Query history
        if st.session_state.query_history:
            st.divider()
            st.title("📜 Query History")
            for i, q in enumerate(st.session_state.query_history[-5:]):
                st.write(f"{i+1}. {q}")
    
    # Main content area
    if st.session_state.db_initialized:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", "7 Companies", "AAPL, MSFT, GOOGL, etc.")
        with col2:
            st.metric("Data Range", "2020-2024", "5 Years")
        with col3:
            st.metric("Tables", "3", "stocks, companies, earnings")
        
        # Query input
        st.divider()
        st.subheader("💬 Ask a Question")
        
        # Example questions
        example_questions = [
            "What is the average closing price of AAPL in 2023?",
            "Which company has the highest revenue?",
            "What was Tesla's EPS in Q1 2023?",
            "How many companies are in the Tech sector?",
            "Compare the average closing price of AAPL and MSFT in 2023",
            "Which stock had the highest trading volume in 2023?"
        ]
        
        selected_example = st.selectbox("Try an example:", [""] + example_questions)
        
        # Text input
        question = st.text_area("Or type your own question:", 
                               value=selected_example if selected_example else "",
                               height=100,
                               placeholder="e.g., What is the total volume for MSFT in 2023?")
        
        # Query button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            run_button = st.button("🚀 Run Query", use_container_width=True)
        
        # Process query
        if run_button and question:
            if not st.session_state.client:
                st.error("❌ Groq API not configured. Please check your API key.")
            else:
                with st.spinner("🔄 Generating SQL and fetching results..."):
                    # Generate SQL
                    sql, error = generate_sql(question, st.session_state.client, schema)
                    
                    if error:
                        st.error(f"❌ Error generating SQL: {error}")
                    else:
                        # Show generated SQL
                        with st.expander("📝 Generated SQL", expanded=True):
                            st.code(sql, language="sql")
                        
                        # Run query
                        result_df, db_error = run_query(sql)
                        
                        if db_error:
                            st.error(f"❌ Database error: {db_error}")
                        else:
                            # Add to history
                            st.session_state.query_history.append(question)
                            
                            # Show results
                            st.success(f"✅ Query executed successfully! Found {len(result_df)} rows.")
                            
                            # Display results in tabs
                            tab1, tab2 = st.tabs(["📊 Results", "📈 Visualization"])
                            
                            with tab1:
                                st.dataframe(result_df, use_container_width=True)
                                
                                # Download button
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download as CSV",
                                    data=csv,
                                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with tab2:
                                fig = create_visualization(result_df, question)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("ℹ️ No visualization available for this data type")
        
        # Data preview section
        with st.expander("🔍 Preview Data Tables"):
            tab1, tab2, tab3 = st.tabs(["Stocks", "Companies", "Earnings"])
            
            with tab1:
                df = get_table_preview("stocks")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Stocks table is being built...")
            
            with tab2:
                df = get_table_preview("companies")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Companies table is being built...")
            
            with tab3:
                df = get_table_preview("earnings")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Earnings table is being built...")
    else:
        st.info("⏳ Database is being initialized. Please wait...")
        with st.spinner("Building database (this will take 1-2 minutes)..."):
            if os.path.exists(DB_PATH):
                st.session_state.db_initialized = True
                st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Built with Streamlit • Powered by Groq Llama 3.3 • Financial Data from Yahoo Finance
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
