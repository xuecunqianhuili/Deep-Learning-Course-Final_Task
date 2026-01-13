from typing import Dict
import numpy as np
import evaluate
import jieba


def build_rouge():
    return evaluate.load("rouge")


def compute_rouge(predictions, references) -> Dict[str, float]:
    rouge = build_rouge()
    # Segment Chinese text with space for ROUGE
    preds_seg = [" ".join(jieba.cut(p)) for p in predictions]
    refs_seg = [" ".join(jieba.cut(r)) for r in references]

    # Use split() as tokenizer since we already segmented with spaces
    results = rouge.compute(predictions=preds_seg, references=refs_seg, tokenizer=lambda x: x.split())
    return {k: float(v) for k, v in results.items()}


def simple_keyword_precision(pred: list, gold: list, k: int = 5) -> float:
    """Compute precision@k for keyword lists (case-insensitive)."""
    pred_k = [p[0].lower() if isinstance(p, (list, tuple)) else str(p).lower() for p in pred[:k]]
    gold_norm = set([str(g).lower() for g in gold])
    hit = sum(1 for x in pred_k if x in gold_norm)
    return hit / max(1, len(pred_k))
