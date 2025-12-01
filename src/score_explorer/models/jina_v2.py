import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class JinaV2Wrapper:
    def __init__(self, model_name="jinaai/jina-reranker-v2-base-multilingual", device=None):
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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def predict(self, query, docs):
        """
        Args:
            query: str
            docs: list of str (document texts)
        Returns:
            list of float scores
        """
        if not docs:
            return []
            
        pairs = [[query, doc] for doc in docs]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()
            
            # If single score, make it a list
            if isinstance(scores, float):
                scores = [scores]
                
        return scores
