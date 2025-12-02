import torch
from transformers import AutoTokenizer, AutoModel

class JinaV3Wrapper:
    def __init__(self, model_name="jinaai/jina-reranker-v3", device=None, dtype=None):
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
            
        # Determine dtype if not provided
        if dtype is None:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif self.device == "cuda":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        print(f"Using dtype: {self.dtype}")

        # V3 uses custom JinaForRanking architecture
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.dtype
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
