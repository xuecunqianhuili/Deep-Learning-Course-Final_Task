from typing import List, Tuple

try:
    from keybert import KeyBERT
except Exception:
    KeyBERT = None


class KeywordGenerator:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if KeyBERT is None:
            raise ImportError("KeyBERT is not installed. Please install keybert and sentence-transformers.")
        print(f"Loading KeyBERT model: {model_name}...")
        self.kb = KeyBERT(model=model_name)
        print("KeyBERT loaded successfully.")

    def generate(self, text: str, top_k: int = 8, diversity: float = 0.6) -> List[Tuple[str, float]]:
        # For Chinese, we can use jieba to tokenize or just let KeyBERT handle it via the model
        # KeyBERT can take a list of candidates. For Chinese, n-grams based on characters or jieba words are better.
        import jieba
        candidates = [" ".join(jieba.lcut(text))] # Simple trick for KeyBERT to handle Chinese
        
        keywords = self.kb.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            # stop_words="chinese", # Need a custom list for Chinese usually
            use_mmr=True,
            diversity=diversity,
            top_n=top_k,
        )
        return keywords

