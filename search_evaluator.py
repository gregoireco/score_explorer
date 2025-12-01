import time
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import plotly.express as px
from tqdm import tqdm

class SearchEvaluator:
    def __init__(self, model_a_predict, model_b_predict):
        """
        Initialize with two prediction functions.
        Each function should accept a query string and return a list of document IDs.
        """
        self.model_a_predict = model_a_predict
        self.model_b_predict = model_b_predict

    def _measure_latency(self, func, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return result, latency_ms

    def _calculate_ndcg(self, predicted_ids, ground_truth_ids, k=10):
        if not ground_truth_ids:
            return 0.0
        
        # Binary relevance: 1 if doc_id in ground_truth, 0 otherwise
        relevance = [1 if doc_id in ground_truth_ids else 0 for doc_id in predicted_ids]
        
        # If fewer than k predictions, pad with 0s
        if len(relevance) < k:
            relevance += [0] * (k - len(relevance))
        
        # We only care about top k
        relevance = relevance[:k]
        
        # Ideal relevance (sorted descending)
        # Since we have binary relevance, ideal is just 1s for the number of relevant docs (up to k)
        num_relevant = min(len(ground_truth_ids), k)
        ideal_relevance = [1] * num_relevant + [0] * (k - num_relevant)
        
        # ndcg_score expects [samples, features]
        # We have 1 sample
        return ndcg_score([ideal_relevance], [relevance], k=k)

    def _calculate_map(self, predicted_ids, ground_truth_ids, k=1000):
        if not ground_truth_ids:
            return 0.0
            
        # Limit predictions to k
        predicted_ids = predicted_ids[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, p in enumerate(predicted_ids):
            if p in ground_truth_ids:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
                
        return score / min(len(ground_truth_ids), k)

    def evaluate(self, queries, ground_truths, k_ndcg=10, k_map=1000):
        """
        Evaluate both models on a list of queries and ground truth document IDs.
        
        Args:
            queries: List of query strings.
            ground_truths: List of sets/lists of relevant document IDs.
            k_ndcg: k for NDCG calculation (default 10). If None, NDCG is skipped.
            k_map: k for MAP calculation (default 1000). If None, MAP is skipped.
            
        Returns:
            pd.DataFrame with evaluation results.
        """
        results = []

        for query, ground_truth in tqdm(zip(queries, ground_truths), total=len(queries), desc="Evaluating queries"):
            ground_truth_set = set(ground_truth)
            
            # Model A
            preds_a, lat_a = self._measure_latency(self.model_a_predict, query)
            
            # Model B
            preds_b, lat_b = self._measure_latency(self.model_b_predict, query)
            
            row = {
                'query': query,
                'model_a_latency_ms': lat_a,
                'model_b_latency_ms': lat_b,
                'latency_diff': lat_a - lat_b
            }
            
            if k_ndcg is not None:
                ndcg_a = self._calculate_ndcg(preds_a, ground_truth_set, k=k_ndcg)
                ndcg_b = self._calculate_ndcg(preds_b, ground_truth_set, k=k_ndcg)
                row.update({
                    'model_a_ndcg': ndcg_a,
                    'model_b_ndcg': ndcg_b,
                    'ndcg_diff': ndcg_a - ndcg_b
                })
                
            if k_map is not None:
                map_a = self._calculate_map(preds_a, ground_truth_set, k=k_map)
                map_b = self._calculate_map(preds_b, ground_truth_set, k=k_map)
                row.update({
                    'model_a_map': map_a,
                    'model_b_map': map_b,
                    'map_diff': map_a - map_b
                })
            
            results.append(row)
            
        return pd.DataFrame(results)

def visualize_results(df, metric='ndcg'):
    """
    Visualize the evaluation results using Plotly.
    X-axis: Latency Difference (A - B)
    Y-axis: Metric Difference (A - B)
    
    Args:
        df: DataFrame containing evaluation results.
        metric: 'ndcg' or 'map' (default 'ndcg').
    """
    metric_col = f'{metric}_diff'
    metric_a = f'model_a_{metric}'
    metric_b = f'model_b_{metric}'
    
    if metric_col not in df.columns:
        raise ValueError(f"Metric column {metric_col} not found in DataFrame")
        
    fig = px.scatter(
        df,
        x='latency_diff',
        y=metric_col,
        hover_data=['query', metric_a, metric_b, 'model_a_latency_ms', 'model_b_latency_ms'],
        title=f'Search Model Comparison ({metric.upper()}): Model A vs Model B',
        labels={
            'latency_diff': 'Latency Difference (ms) (Positive = A is slower)',
            metric_col: f'{metric.upper()} Difference (Positive = A is better)'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig
