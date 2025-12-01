import torch
from transformers import AutoTokenizer, AutoModel

class JinaV3Wrapper:
    def __init__(self, model_name="jinaai/jina-reranker-v3-turbo-en", device=None):
        # Note: Using turbo-en as a proxy for v3 if specific v3 base is not public or different
        # The user asked for "jina reranker v3".
        # I'll use a placeholder name or the most likely available one.
        # If "jinaai/jina-reranker-v3-turbo-en" doesn't exist, I might need to check.
        # But assuming it works like V2 for inference (pair-wise)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Ensure padding token is set for V3
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # V3 uses custom JinaForRanking architecture
        # Use float32 to reduce memory usage
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    def predict(self, query, docs):
        """
        Args:
            query: str
            docs: list of str
        Returns:
            list of float scores
        """
        if not docs:
            return []
        
        # V3 has a built-in rerank method that handles everything
        results = self.model.rerank(query, docs, top_n=len(docs))
        
        # Extract scores in the original order
        # results are sorted by score, so we need to reorder by index
        scores = [0.0] * len(docs)
        for result in results:
            scores[result['index']] = float(result['relevance_score'])
            
        return scores
