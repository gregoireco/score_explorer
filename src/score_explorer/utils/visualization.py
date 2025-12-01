import plotly.express as px

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
    
    # Prepare hover data
    hover_data = {
        'query': True,
        metric_a: ':.4f',
        metric_b: ':.4f',
        'model_a_latency_ms': ':.2f',
        'model_b_latency_ms': ':.2f',
        'latency_diff': ':.2f',
        metric_col: ':.4f'
    }
    
    # Add optional columns if present
    if 'query_text' in df.columns:
        # Truncate long queries for display
        df['query_text_short'] = df['query_text'].apply(lambda x: (x[:100] + '...') if isinstance(x, str) and len(x) > 100 else x)
        hover_data['query_text_short'] = True
        
    model_a_name = df['model_a_name'].iloc[0] if 'model_a_name' in df.columns else "Model A"
    model_b_name = df['model_b_name'].iloc[0] if 'model_b_name' in df.columns else "Model B"
        
    fig = px.scatter(
        df,
        x='latency_diff',
        y=metric_col,
        hover_data=hover_data,
        title=f'Search Model Comparison ({metric.upper()}): {model_a_name} vs {model_b_name}',
        labels={
            'latency_diff': f'Latency Difference (ms) (Positive = {model_a_name} is slower)',
            metric_col: f'{metric.upper()} Difference (Positive = {model_a_name} is better)'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig
