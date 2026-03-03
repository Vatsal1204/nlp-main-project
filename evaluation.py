import sqlite3
import pandas as pd
from engine import run_query, ask_question, setup_groq_client
import time
from tabulate import tabulate

DB_PATH = "data/finance.db"

# Expanded test dataset with more variety
test_queries = [
    # Easy queries
    {
        "question": "What is the average closing price of AAPL in 2023?",
        "expected_sql": "SELECT AVG(Close) as avg_close FROM stocks WHERE Ticker = 'AAPL' AND strftime('%Y', Date) = '2023'",
        "difficulty": "easy",
        "category": "aggregation",
        "description": "Simple year filter with aggregation"
    },
    {
        "question": "Which company has the highest revenue?",
        "expected_sql": "SELECT Company, Revenue_B FROM companies ORDER BY Revenue_B DESC LIMIT 1",
        "difficulty": "easy",
        "category": "sorting",
        "description": "Simple ORDER BY with LIMIT"
    },
    {
        "question": "How many companies are in the Tech sector?",
        "expected_sql": "SELECT COUNT(*) as count FROM companies WHERE Sector = 'Tech'",
        "difficulty": "easy",
        "category": "aggregation",
        "description": "COUNT with WHERE clause"
    },
    
    # Medium queries
    {
        "question": "What was Tesla's EPS in Q1 2023?",
        "expected_sql": "SELECT EPS FROM earnings WHERE Ticker = 'TSLA' AND Quarter = 'Q1-2023'",
        "difficulty": "medium",
        "category": "filtering",
        "description": "Simple filter with two conditions"
    },
    {
        "question": "Show me all stocks with closing price above $400 in 2024",
        "expected_sql": "SELECT * FROM stocks WHERE Close > 400 AND strftime('%Y', Date) = '2024'",
        "difficulty": "medium",
        "category": "filtering",
        "description": "Numeric and date filtering"
    },
    {
        "question": "What is the total trading volume for MSFT in 2023?",
        "expected_sql": "SELECT SUM(Volume) as total_volume FROM stocks WHERE Ticker = 'MSFT' AND strftime('%Y', Date) = '2023'",
        "difficulty": "medium",
        "category": "aggregation",
        "description": "SUM with multiple conditions"
    },
    {
        "question": "What is the maximum closing price of TSLA in 2024?",
        "expected_sql": "SELECT MAX(Close) as max_close FROM stocks WHERE Ticker = 'TSLA' AND strftime('%Y', Date) = '2024'",
        "difficulty": "medium",
        "category": "aggregation",
        "description": "MAX with filtering"
    },
    
    # Hard queries
    {
        "question": "Which quarter had the highest revenue for Apple in 2023?",
        "expected_sql": "SELECT Quarter, Revenue_B FROM earnings WHERE Ticker = 'AAPL' AND Quarter LIKE '%2023' ORDER BY Revenue_B DESC LIMIT 1",
        "difficulty": "hard",
        "category": "complex_filtering",
        "description": "Complex date pattern matching with ordering"
    },
    {
        "question": "Compare the average closing price of AAPL and MSFT in 2023",
        "expected_sql": "SELECT Ticker, AVG(Close) as avg_close FROM stocks WHERE Ticker IN ('AAPL', 'MSFT') AND strftime('%Y', Date) = '2023' GROUP BY Ticker",
        "difficulty": "hard",
        "category": "group_by",
        "description": "GROUP BY with multiple values"
    },
    {
        "question": "Which stock had the highest trading volume in 2023?",
        "expected_sql": "SELECT Ticker, SUM(Volume) as total_volume FROM stocks WHERE strftime('%Y', Date) = '2023' GROUP BY Ticker ORDER BY total_volume DESC LIMIT 1",
        "difficulty": "hard",
        "category": "complex_aggregation",
        "description": "GROUP BY with ORDER BY and LIMIT"
    },
    {
        "question": "What is the total revenue of all Tech sector companies?",
        "expected_sql": "SELECT SUM(Revenue_B) as total_revenue FROM companies WHERE Sector = 'Tech'",
        "difficulty": "hard",
        "category": "aggregation",
        "description": "SUM with JOIN-like condition (single table)"
    }
]

def execution_accuracy(expected_sql, predicted_sql, tolerance=True):
    """Compare execution results with tolerance for column naming"""
    try:
        expected_result = run_query(expected_sql)[0]
        predicted_result = run_query(predicted_sql)[0]
        
        if expected_result is None or predicted_result is None:
            return False, "Query execution failed"
        
        # For count queries, compare values even if column names differ
        if 'COUNT' in expected_sql.upper() and 'COUNT' in predicted_sql.upper():
            return expected_result.iloc[0,0] == predicted_result.iloc[0,0], "Value match (count)"
        
        # For aggregation with aliases, compare values
        if len(expected_result.columns) == 1 and len(predicted_result.columns) == 1:
            return expected_result.iloc[0,0] == predicted_result.iloc[0,0], "Value match (single column)"
        
        # For multi-column results, try to align by position
        if len(expected_result.columns) == len(predicted_result.columns):
            try:
                # Try positional comparison
                comparison = (expected_result.values == predicted_result.values).all()
                if comparison:
                    return True, "Positional match"
            except:
                pass
        
        # Full DataFrame comparison
        try:
            return expected_result.equals(predicted_result), "Full DataFrame match"
        except:
            return False, "Structure mismatch"
            
    except Exception as e:
        return False, f"Comparison error: {str(e)}"

def evaluate_model(use_few_shot=True):
    """Run evaluation on all test queries"""
    print("🔬 Starting Model Evaluation...")
    print("=" * 80)
    print(f"Mode: {'With few-shot' if use_few_shot else 'Without few-shot'}")
    print("=" * 80)
    
    client = setup_groq_client()
    if not client:
        return
    
    results = {
        "total": len(test_queries),
        "correct": 0,
        "by_difficulty": {"easy": {"total": 0, "correct": 0},
                          "medium": {"total": 0, "correct": 0},
                          "hard": {"total": 0, "correct": 0}},
        "by_category": {},
        "errors": [],
        "details": []
    }
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}/{len(test_queries)}: {test['question']}")
        print(f"   Difficulty: {test['difficulty']} | Category: {test['category']}")
        print(f"   Description: {test['description']}")
        
        # Get prediction
        start_time = time.time()
        result_df, predicted_sql = ask_question(test['question'], client)
        elapsed = time.time() - start_time
        
        if predicted_sql:
            # Check accuracy
            is_correct, match_type = execution_accuracy(test['expected_sql'], predicted_sql)
            
            # Update counts
            results["by_difficulty"][test["difficulty"]]["total"] += 1
            
            # Track by category
            cat = test["category"]
            if cat not in results["by_category"]:
                results["by_category"][cat] = {"total": 0, "correct": 0}
            results["by_category"][cat]["total"] += 1
            
            # Store detail
            detail = {
                "question": test['question'],
                "difficulty": test['difficulty'],
                "category": test['category'],
                "correct": is_correct,
                "time": elapsed,
                "match_type": match_type
            }
            
            if is_correct:
                results["correct"] += 1
                results["by_difficulty"][test["difficulty"]]["correct"] += 1
                results["by_category"][cat]["correct"] += 1
                print(f"   ✅ CORRECT ({elapsed:.2f}s) - {match_type}")
            else:
                results["errors"].append({
                    "question": test['question'],
                    "expected": test['expected_sql'],
                    "predicted": predicted_sql
                })
                print(f"   ❌ INCORRECT ({elapsed:.2f}s)")
                print(f"      Expected: {test['expected_sql']}")
                print(f"      Got: {predicted_sql}")
            
            results["details"].append(detail)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    
    total_acc = results['correct']/results['total']*100
    print(f"🎯 Overall Accuracy: {results['correct']}/{results['total']} ({total_acc:.1f}%)")
    
    print("\n📈 By Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        stats = results["by_difficulty"][diff]
        if stats["total"] > 0:
            acc = stats["correct"]/stats["total"]*100
            print(f"  {diff.capitalize()}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    print("\n📊 By Category:")
    categories_data = []
    for cat, stats in results["by_category"].items():
        acc = stats["correct"]/stats["total"]*100
        categories_data.append([cat, stats["correct"], stats["total"], f"{acc:.1f}%"])
    
    print(tabulate(categories_data, headers=["Category", "Correct", "Total", "Accuracy"], tablefmt="grid"))
    
    # Performance by time
    avg_time = sum(d['time'] for d in results['details']) / len(results['details'])
    print(f"\n⏱️  Average response time: {avg_time:.2f}s")
    
    if results["errors"]:
        print(f"\n❌ Failed Queries ({len(results['errors'])}):")
        for err in results["errors"]:
            print(f"  • {err['question']}")
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "fewshot" if use_few_shot else "base"
    filename = f"evaluation_results_{mode}_{timestamp}.csv"
    
    df_results = pd.DataFrame(results["details"])
    df_results.to_csv(filename, index=False)
    print(f"\n💾 Detailed results saved to: {filename}")
    
    return results

def compare_models():
    """Compare performance with and without few-shot learning"""
    print("🔄 Comparing Few-shot vs Base performance")
    print("=" * 80)
    
    print("\n📊 Running evaluation WITHOUT few-shot...")
    base_results = evaluate_model(use_few_shot=False)
    
    print("\n📊 Running evaluation WITH few-shot...")
    fewshot_results = evaluate_model(use_few_shot=True)
    
    if base_results and fewshot_results:
        print("\n" + "=" * 80)
        print("📈 COMPARISON SUMMARY")
        print("=" * 80)
        
        base_acc = base_results['correct']/base_results['total']*100
        fewshot_acc = fewshot_results['correct']/fewshot_results['total']*100
        
        comparison = [
            ["Base (no few-shot)", f"{base_acc:.1f}%", base_results['correct'], base_results['total']],
            ["Few-shot", f"{fewshot_acc:.1f}%", fewshot_results['correct'], fewshot_results['total']],
            ["Improvement", f"{fewshot_acc - base_acc:+.1f}%", 
             fewshot_results['correct'] - base_results['correct'], ""]
        ]
        
        print(tabulate(comparison, headers=["Method", "Accuracy", "Correct", "Total"], tablefmt="grid"))

if __name__ == "__main__":
    print("🔬 Advanced Evaluation Suite")
    print("=" * 80)
    
    print("Select evaluation mode:")
    print("1. Run standard evaluation (with few-shot)")
    print("2. Run evaluation without few-shot")
    print("3. Compare with/without few-shot")
    print("4. Run specific category tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        evaluate_model(use_few_shot=True)
    elif choice == "2":
        evaluate_model(use_few_shot=False)
    elif choice == "3":
        compare_models()
    elif choice == "4":
        client = setup_groq_client()
        if client:
            print("\nAvailable categories:")
            categories = set(q['category'] for q in test_queries)
            for cat in categories:
                print(f"  • {cat}")
            
            cat_choice = input("\nEnter category to test: ").strip()
            filtered_queries = [q for q in test_queries if q['category'] == cat_choice]
            
            for test in filtered_queries:
                print(f"\n📝 {test['question']}")
                ask_question(test['question'], client)
                print("-" * 50)