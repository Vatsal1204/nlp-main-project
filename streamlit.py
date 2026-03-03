import streamlit as st
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration
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
    .warning-box {
        padding: 1rem;
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
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

# Database path
DB_PATH = "data/finance.db"

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'client' not in st.session_state:
    st.session_state.client = None

# Database functions
@st.cache_data(ttl=3600)
def get_db_schema():
    """Get database schema for context"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    schema = ""
    tables = ["stocks", "companies", "earnings"]
    table_info = {}
    
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        col_types = [col[2] for col in columns]
        
        schema += f"Table '{table}':\n"
        table_info[table] = []
        for name, type_ in zip(col_names, col_types):
            schema += f"  - {name} ({type_})\n"
            table_info[table].append({"name": name, "type": type_})
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        schema += f"  → {count} rows\n\n"
        table_info[table].append({"row_count": count})
    
    conn.close()
    return schema, table_info

@st.cache_data(ttl=300)
def get_table_preview(table_name, limit=10):
    """Get preview of table data"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
    conn.close()
    return df

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
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

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

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Financial NLP Query Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about stock data in plain English</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stocks.png", width=100)
        st.title("📁 Database Info")
        
        # Check if database exists
        if not os.path.exists(DB_PATH):
            st.error("❌ Database not found! Please run dataset.py first.")
            st.info("Run: `python dataset.py` to create the database.")
            return
        
        # Load schema
        with st.spinner("Loading database schema..."):
            schema, table_info = get_db_schema()
        
        # Display table info
        for table, info in table_info.items():
            with st.expander(f"📋 {table.upper()}"):
                cols = [col for col in info if isinstance(col, dict)]
                for col in cols[:5]:  # Show first 5 columns
                    st.write(f"• {col['name']} ({col['type']})")
                row_count = next((item for item in info if 'row_count' in item), None)
                if row_count:
                    st.write(f"**Rows:** {row_count['row_count']:,}")
        
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
            return
        
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
            st.dataframe(get_table_preview("stocks"), use_container_width=True)
        
        with tab2:
            st.dataframe(get_table_preview("companies"), use_container_width=True)
        
        with tab3:
            st.dataframe(get_table_preview("earnings"), use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Built with Streamlit • Powered by Groq Llama 3.3 • Financial Data from Yahoo Finance
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()