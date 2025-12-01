import pandas as pd
from data_loader import DataLoader
from models import JinaV2Wrapper, JinaV3Wrapper
from search_evaluator import SearchEvaluator, visualize_results
import torch

def main():
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader(task_name="mteb/AlloprofReranking", split="test")
    loader.load_data()
    
    # Use a subset of queries for faster evaluation if needed
    # For full evaluation, remove the slice
    query_ids = list(loader.queries.keys())[:5] 
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
        model_b = JinaV3Wrapper(model_name="./jina-reranker-v3", device="cpu")
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
    
    # Add actual query text for readability
    df['query_text'] = df['query'].apply(lambda qid: loader.get_query_text(qid))
    
    print("\nEvaluation Results:")
    print(df.head())
    print(f"Mean MAP Diff: {df['map_diff'].mean()}")
    
    # 4. Save Results
    df.to_csv("evaluation_results.csv", index=False)
    
    fig_map = visualize_results(df, metric='map')
    fig_map.write_html("evaluation_results_map.html")
    print("Results saved to evaluation_results.csv and evaluation_results_map.html")

if __name__ == "__main__":
    main()
