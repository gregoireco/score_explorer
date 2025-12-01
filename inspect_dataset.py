from mteb import MTEB

try:
    print("Attempting to load AlloprofReranking using MTEB...")
    mteb = MTEB(tasks=["AlloprofReranking"])
    task = mteb.tasks[0]
    task.load_data()
    
    print("Data loaded successfully.")
    if task.corpus:
        print("Corpus available.")
        print("First corpus item:", list(task.corpus['test'].items())[0] if 'test' in task.corpus else "No test corpus")
    
    if task.queries:
        print("Queries available.")
        print("First query item:", list(task.queries['test'].items())[0] if 'test' in task.queries else "No test queries")
        
    if task.relevant_docs:
        print("Relevant docs available.")
        print("First relevant doc item:", list(task.relevant_docs['test'].items())[0] if 'test' in task.relevant_docs else "No test relevant docs")

except Exception as e:
    print(f"Error loading with MTEB: {e}")
