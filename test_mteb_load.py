from mteb import get_tasks

try:
    print("Getting tasks...")
    tasks = get_tasks(tasks=["AlloprofReranking"])
    print(f"Got {len(tasks)} tasks.")
    task = tasks[0]
    print(f"Task object: {task}")
    
    print("Loading data...")
    task.load_data()
    print("Data loaded.")
    
    print("Data loaded.")
    print("Task attributes:", dir(task))
    print("Trying manual loading...")
    from datasets import load_dataset
    
    try:
        corpus = load_dataset("mteb/AlloprofReranking", data_files={"corpus": "corpus/*.parquet"})
        print("Manual corpus load success:", corpus)
        print("Corpus features:", corpus['corpus'].features)
        print("First corpus item:", corpus['corpus'][0])
    except Exception as e:
        print(f"Manual corpus load failed: {e}")
        
    try:
        queries = load_dataset("mteb/AlloprofReranking", data_files={"queries": "queries/*.jsonl"}) # Guessing extension
        print("Manual queries load success:", queries)
    except Exception as e:
        print(f"Manual queries load failed (jsonl): {e}")
        try:
            queries = load_dataset("mteb/AlloprofReranking", data_files={"queries": "queries/*.parquet"})
            print("Manual queries load success (parquet):", queries)
        except Exception as e2:
             print(f"Manual queries load failed (parquet): {e2}")

except Exception as e:
    print(f"Error: {e}")
