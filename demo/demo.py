import argparse
import json
import os
import sys
import torch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model.keyword_generator import KeywordGenerator


def summarize_text(model_dir: str, text: str, max_length: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Summarizing on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    
    # mT5 fine-tuned on XLSum typically uses raw text input without prefix
    inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs, 
        max_length=max_length, 
        num_beams=4, 
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        early_stopping=True
    )
    summary = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return summary


def run_demo(model_dir: str, text: str, top_k: int, diversity: float, out_path: str):
    summary = summarize_text(model_dir, text)
    kg = KeywordGenerator()
    keywords = kg.generate(text, top_k=top_k, diversity=diversity)

    result = {
        "summary": summary,
        "keywords": keywords,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run summarization + keyword generation demo")
    parser.add_argument("--model_dir", type=str, default=os.path.join("results", "summarizer_mt5_small"))
    parser.add_argument("--text", type=str, required=True, help="Input text to summarize and extract keywords")

    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--diversity", type=float, default=0.6)
    parser.add_argument("--out", type=str, default=os.path.join("results", "demo_output.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.model_dir, args.text, args.top_k, args.diversity, args.out)
