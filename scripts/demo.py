import time
import random
from score_explorer.evaluation import SearchEvaluator
from score_explorer.utils import visualize_results

# Mock Data
QUERIES = [
    "python framework",
    "machine learning",
    "search engine",
    "latency optimization",
    "data science"
]

# Ground Truth (doc_ids)
GROUND_TRUTHS = [
    {"doc_1", "doc_2", "doc_3"},
    {"doc_4", "doc_5"},
    {"doc_6", "doc_7", "doc_8", "doc_9"},
    {"doc_10"},
    {"doc_11", "doc_12", "doc_13"}
]

# Mock Models
def model_a_predict(query):
    # Fast but random
    time.sleep(random.uniform(0.01, 0.05))
    # Return random doc ids from a pool
    return [f"doc_{random.randint(1, 20)}" for _ in range(10)]

def model_b_predict(query):
    # Slower but "smarter" (cheating by looking at ground truth for some queries)
    time.sleep(random.uniform(0.1, 0.3))
    
    # Cheat for specific queries to show better performance
    if "python" in query or "search" in query:
        # Return some correct docs mixed with noise
        # Find the index of the query to get ground truth (simplified for demo)
        try:
            idx = QUERIES.index(query)
            gt = list(GROUND_TRUTHS[idx])
            return gt + [f"doc_{random.randint(20, 30)}" for _ in range(10 - len(gt))]
        except ValueError:
            pass
            
    return [f"doc_{random.randint(1, 20)}" for _ in range(10)]

def main():
    print("Initializing SearchEvaluator...")
    evaluator = SearchEvaluator(model_a_predict, model_b_predict)
    
    print("Running evaluation (NDCG@10, MAP@1000)...")
    df = evaluator.evaluate(QUERIES, GROUND_TRUTHS, k_ndcg=10, k_map=1000)
    
    print("\nEvaluation Results:")
    print(df)
    
    print("\nGenerating visualizations...")
    
    # NDCG Plot
    fig_ndcg = visualize_results(df, metric='ndcg')
    fig_ndcg.write_html("evaluation_results_ndcg.html")
    print("NDCG visualization saved to evaluation_results_ndcg.html")
    
    # MAP Plot
    fig_map = visualize_results(df, metric='map')
    fig_map.write_html("evaluation_results_map.html")
    print("MAP visualization saved to evaluation_results_map.html")

if __name__ == "__main__":
    main()
