import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()

DB_PATH = "data/finance.db"

def get_db_schema(include_samples=True):
    """Get database schema for context"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    schema = ""
    tables = ["stocks", "companies", "earnings"]
    
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        col_types = [col[2] for col in columns]
        
        # Add column info with types
        schema += f"Table '{table}':\n"
        for name, type_ in zip(col_names, col_types):
            schema += f"  - {name} ({type_})\n"
        
        # Get sample data
        if include_samples:
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            samples = cursor.fetchall()
            if samples:
                schema += f"  Sample rows: {samples}\n"
        schema += "\n"
    
    conn.close()
    return schema

def setup_groq_client():
    """Setup Groq client"""
    print("🤖 Setting up Groq Client...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file")
        print("Please add: GROQ_API_KEY=your_key_here")
        return None
    
    client = Groq(api_key=api_key)
    print("✅ Groq Client ready!")
    return client

def clean_sql(sql):
    """Clean SQL query returned by LLM"""
    sql = sql.strip()
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.replace("SQLQuery:", "").strip()
    sql = sql.replace("SQL Query:", "").strip()
    sql = sql.replace(";", "")  # Remove semicolon at end
    return sql

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

def ask_question(question, client, model="llama-3.3-70b-versatile", use_few_shot=True):
    """Convert natural language to SQL using Groq with few-shot examples"""
    print(f"\n❓ Question: {question}")
    
    # Get database schema
    schema = get_db_schema(include_samples=True)
    
    # Few-shot examples for complex patterns
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
    
    # Base prompt
    base_prompt = f"""You are a SQLite expert. Generate a SQL query to answer the question.

Database Schema:
{schema}

Important rules:
- Use proper SQLite syntax
- Always use single quotes for strings
- For year filtering: strftime('%Y', Date) = 'YYYY'
- For quarter filtering: Quarter LIKE '%YYYY'
- Always include meaningful column aliases
- Return ONLY the SQL query, no explanations
- Do NOT include semicolon at the end

Question: {question}

SQL Query:"""
    
    # Few-shot prompt
    few_shot_prompt = f"""You are a SQLite expert. Generate a SQL query to answer the question.

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
- Do NOT include semicolon at the end

Question: {question}

SQL Query:"""
    
    # Select prompt based on parameter
    prompt = few_shot_prompt if use_few_shot else base_prompt
    
    try:
        # Generate SQL using Groq
        print("🤔 Thinking...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate only the SQL query, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        sql = clean_sql(response.choices[0].message.content)
        print(f"📝 Generated SQL: {sql}")
        
        # Run the SQL
        result_df, error = run_query(sql)
        
        if error:
            print(f"❌ Error: {error}")
            return None, sql
        else:
            print(f"✅ Result:")
            if len(result_df) > 0:
                print(result_df.to_string())
            else:
                print("No results found")
            return result_df, sql
    
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return None, None

def test_database_connection():
    """Test if we can connect to the database and run a simple query"""
    print("🧪 Testing database connection...")
    try:
        test_query = "SELECT COUNT(*) as count FROM stocks"
        result, error = run_query(test_query)
        if error:
            print(f"❌ Database test failed: {error}")
            return False
        else:
            count = result['count'].iloc[0]
            print(f"✅ Database connected. Found {count} stock records")
            return True
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def interactive_mode(client):
    """Run interactive query mode"""
    print("\n🎯 Interactive Mode (type 'exit' to quit)")
    print("=" * 50)
    
    while True:
        question = input("\n💬 Your question: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            break
        if not question:
            continue
        
        ask_question(question, client)
        print("-" * 50)

if __name__ == "__main__":
    print("🚀 Advanced NLP Engine with Groq (FREE)")
    print("=" * 60)
    
    # Test database connection first
    if not test_database_connection():
        print("❌ Cannot connect to database. Please run dataset.py first.")
        exit(1)
    
    # Setup Groq client
    client = setup_groq_client()
    if not client:
        exit(1)
    
    # Ask user for mode
    print("\n📌 Select mode:")
    print("1. Run test queries")
    print("2. Interactive mode")
    print("3. Compare with/without few-shot")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Test questions
        test_questions = [
            "What is the average closing price of AAPL in 2023?",
            "Which company has the highest revenue?",
            "What was Tesla's EPS in Q1 2023?",
            "How many companies are in the Tech sector?",
            "Which quarter had the highest revenue for Apple in 2023?",
            "What is the total volume for MSFT in 2023?",
        ]
        
        for question in test_questions:
            ask_question(question, client)
            print("-" * 60)
    
    elif choice == "2":
        interactive_mode(client)
    
    elif choice == "3":
        print("\n🔬 Comparing with and without few-shot examples:")
        test_questions = [
            "Which quarter had the highest revenue for Apple in 2023?",
            "How many companies are in the Tech sector?",
        ]
        
        for question in test_questions:
            print("\n" + "=" * 60)
            print(f"📋 Question: {question}")
            print("-" * 60)
            print("🔄 Without few-shot:")
            ask_question(question, client, use_few_shot=False)
            print("-" * 30)
            print("🔄 With few-shot:")
            ask_question(question, client, use_few_shot=True)
            print("=" * 60)