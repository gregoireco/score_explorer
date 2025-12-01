import pandas as pd
import torch
import argparse
from datetime import datetime
from pathlib import Path
from score_explorer.data import DataLoader
from score_explorer.models import JinaV2Wrapper, JinaV3Wrapper
from score_explorer.evaluation import SearchEvaluator
from score_explorer.utils import visualize_results

# Get project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

def main(num_queries=None, test_mode=False):
    """
    Run comparison between Jina V2 and V3 rerankers.
    
    Args:
        num_queries: Number of queries to evaluate (None = all queries)
        test_mode: If True, run quick test with 3 queries
    """
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader(task_name="mteb/AlloprofReranking", split="test")
    loader.load_data()
    
    # Determine number of queries and mode to evaluate
    if test_mode and num_queries:
        selected_num_queries = num_queries
        print(f"📊 Running test evaluation on {selected_num_queries} queries")
    elif test_mode and num_queries is None:
        selected_num_queries = 5
        print(f"📊 Running test evaluation on {selected_num_queries} queries")
    else:
        selected_num_queries = -1
        print(f"📊 Running evaluation on all queries")
    
    query_ids = list(loader.queries.keys())[:selected_num_queries]
    print(f"Evaluating on {len(query_ids)} queries.")
    
    queries = [loader.get_query_text(qid) for qid in query_ids]
    ground_truths = [loader.relevant_docs.get(qid, {}) for qid in query_ids]
    
    print(f"Loaded {len(loader.queries)} queries in total.")
    print(f"Selected {len(query_ids)} queries for evaluation.")
    if not query_ids:
        print("WARNING: No queries selected! Check data loading.")
        return
    
    # 2. Initialize Models
    print("Initializing models...")
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Using device: {device}")
    
    # Model A: Jina V2
    model_a = JinaV2Wrapper(device=device)
    
    # Model B: Jina V3
    try:
        # Use local V3 weights
        # V3 has very large context window (131k) - use CPU to avoid MPS OOM
        model_b = JinaV3Wrapper(model_name=str(MODELS_DIR / "jina-reranker-v3"), device="cpu")
    except Exception as e:
        print(f"Failed to load V3 (./jina-reranker-v3): {e}")
        print("Falling back to V2 for Model B (Comparison will be V2 vs V2)")
        model_b = JinaV2Wrapper(device=device)

    # Define prediction functions that match SearchEvaluator interface
    # SearchEvaluator expects: func(query) -> list of doc_ids
    # But our models need candidates.
    # So we wrap them to fetch candidates first.
    
    def predict_with_model(model, query, qid):
        # 1. Get candidates (BM25)
        candidate_ids = loader.get_candidates(qid, k=10) # Further reduced for V3 memory constraints
        candidate_docs = [loader.get_doc_text(doc_id) for doc_id in candidate_ids]
        
        # 2. Rerank
        scores = model.predict(query, candidate_docs)
        
        # 3. Sort candidates by score
        scored_candidates = list(zip(candidate_ids, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [doc_id for doc_id, score in scored_candidates]

    # We need to bind qid to the function, but SearchEvaluator passes only query
    # We can pass a list of (query, qid) tuples to evaluate?
    # Or modify SearchEvaluator to handle this?
    # Or simply use a closure if we iterate manually?
    
    # SearchEvaluator.evaluate takes lists of queries and ground_truths
    # It iterates: for query, ground_truth in zip(queries, ground_truths)
    # And calls: self.model_a_predict(query)
    
    # Problem: model_a_predict needs qid to get candidates.
    # Solution: Pass qid as the "query" to the evaluator, and have the wrapper handle it.
    
    def model_a_wrapper(qid):
        query_text = loader.get_query_text(qid)
        return predict_with_model(model_a, query_text, qid)
        
    def model_b_wrapper(qid):
        query_text = loader.get_query_text(qid)
        return predict_with_model(model_b, query_text, qid)
        
    # 3. Run Evaluation
    print("Running evaluation...")
    evaluator = SearchEvaluator(model_a_wrapper, model_b_wrapper)
    
    # Pass query_ids as "queries"
    df = evaluator.evaluate(query_ids, ground_truths, k_ndcg=10, k_map=1000)
    
    # Add metadata
    df['query_text'] = df['query'].apply(lambda qid: loader.get_query_text(qid))
    df['model_a_name'] = "Jina V2"
    df['model_b_name'] = "Jina V3"
    
    print("\nEvaluation Results:")
    print(df.head())
    print(f"Mean MAP Diff: {df['map_diff'].mean()}")
    
    # 4. Save Results with Run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_n{selected_num_queries}"
    
    csv_filename = f"evaluation_results_{run_id}.csv"
    html_filename = f"evaluation_results_map_{run_id}.html"
    
    df.to_csv(RESULTS_DIR / csv_filename, index=False)
    
    fig_map = visualize_results(df, metric='map')
    fig_map.write_html(RESULTS_DIR / html_filename)
    print(f"Results saved to results/{csv_filename} and results/{html_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Jina Reranker V2 and V3 on AlloProf dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 3 queries
  python scripts/run_comparison.py --test
  
  # Evaluate on 10 queries
  python scripts/run_comparison.py -n 10
  
  # Full evaluation on all queries
  python scripts/run_comparison.py --all
        """
    )
    
    parser.add_argument(
        "-n", "--num-queries",
        type=int,
        help="Number of queries to evaluate"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test mode (defaults to 3 queries, or use -n to specify)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full evaluation on all queries (~2316 queries, may take hours)"
    )
    
    args = parser.parse_args()
    
    # Determine parameters
    if args.all:
        num_queries = None
        test_mode = False
    elif args.test:
        test_mode = True
        # If -n is provided with --test, use it. Otherwise default to 3.
        if args.num_queries:
            num_queries = args.num_queries
        else:
            num_queries = 3
    elif args.num_queries:
        num_queries = args.num_queries
        test_mode = False
    else:
        # Default: all queries
        num_queries = None
        test_mode = False
        print("ℹ️  No arguments provided, defaulting to all queries. Use --help for options.")
    
    main(num_queries=num_queries, test_mode=test_mode)
