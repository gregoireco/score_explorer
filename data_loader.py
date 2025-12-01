import random
from mteb import MTEB
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm

class DataLoader:
    def __init__(self, task_name="AlloprofReranking", split="test"):
        self.task_name = task_name
        self.split = split
        self.mteb = MTEB(tasks=[task_name])
        self.task = self.mteb.tasks[0]
        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}
        self.bm25 = None
        self.corpus_ids = []

    def load_data(self):
        """Loads data from Hugging Face directly."""
        print(f"Loading {self.task_name} dataset...")
        from datasets import load_dataset
        
        # 1. Load Qrels (Default config)
        # This usually contains the qrels (query-id, corpus-id, score)
        try:
            qrels_ds = load_dataset(self.task_name, trust_remote_code=True)
            if self.split in qrels_ds:
                self.relevant_docs = {}
                for row in tqdm(qrels_ds[self.split], desc="Processing qrels"):
                    qid = row['query-id']
                    doc_id = row['corpus-id']
                    score = row['score']
                    if qid not in self.relevant_docs:
                        self.relevant_docs[qid] = {}
                    self.relevant_docs[qid][doc_id] = score
            else:
                print(f"Split {self.split} not found in qrels.")
        except Exception as e:
            print(f"Error loading qrels: {e}")

        # 2. Load Corpus
        try:
            corpus_ds = load_dataset(self.task_name, data_files={"corpus": "corpus/*.parquet"})
            self.corpus = {}
            # Corpus usually has '_id' and 'text'
            for row in tqdm(corpus_ds['corpus'], desc="Processing corpus"):
                self.corpus[row['_id']] = {"text": row['text']}
        except Exception as e:
            print(f"Error loading corpus: {e}")

        # 3. Load Queries
        try:
            queries_ds = load_dataset(self.task_name, data_files={"queries": "queries/*.parquet"})
            self.queries = {}
            # Queries usually has '_id' and 'text'
            for row in tqdm(queries_ds['queries'], desc="Processing queries"):
                self.queries[row['_id']] = row['text']
        except Exception as e:
            print(f"Error loading queries: {e}")
            
        print(f"Loaded {len(self.corpus)} documents, {len(self.queries)} queries, and {len(self.relevant_docs)} qrels.")

    def prepare_bm25(self):
        """Indexes the corpus using BM25."""
        print("Indexing corpus with BM25...")
        self.corpus_ids = list(self.corpus.keys())
        tokenized_corpus = [self.corpus[doc_id].get("text", "").split(" ") for doc_id in self.corpus_ids]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 indexing complete.")

    def get_candidates(self, query_id, k=100):
        """
        Returns candidate document IDs for a given query.
        If candidates are not pre-computed, uses BM25 to retrieve them.
        """
        # Check if task has top_ranked (candidates)
        # MTEB tasks might not have this loaded by default in the same way
        # For this task, we'll assume we need to generate them or they are not easily accessible via standard MTEB load
        
        if self.bm25 is None:
            self.prepare_bm25()
            
        query_text = self.queries[query_id]
        tokenized_query = query_text.split(" ")
        
        # Get top-k scores
        # BM25Okapi.get_top_n returns the documents, but we need IDs
        # So we use get_scores
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:k]
        
        candidate_ids = [self.corpus_ids[i] for i in top_n_indices]
        return candidate_ids

    def get_query_text(self, query_id):
        return self.queries.get(query_id, "")

    def get_doc_text(self, doc_id):
        return self.corpus.get(doc_id, {}).get("text", "")
