import argparse
import os
import sys
import torch
import jieba
import json
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data.load_datasets import load_summarization_dataset

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def compute_metrics(predictions, references):
    rouge = load_metric("rouge", trust_remote_code=True)
    
    # Jieba cut for Chinese ROUGE
    # rouge expects tokens separated by space
    pred_tokens = [" ".join(jieba.cut(p)) for p in predictions]
    ref_tokens = [" ".join(jieba.cut(r)) for r in references]
    
    # We use rouge-1, rouge-2, rouge-l
    results = rouge.compute(predictions=pred_tokens, references=ref_tokens)
    
    # Extract f-measure
    final_results = {}
    for key, value in results.items():
        final_results[key] = value.mid.fmeasure * 100
    
    return final_results

def evaluate_model(model_dir, subset_size=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}...")

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    # Load Test Data
    print("Loading test dataset...")
    ds = load_summarization_dataset("xlsum_zh", subset_size=subset_size)
    test_data = ds["test"]
    
    predictions = []
    references = []

    print("Running inference...")
    for batch in tqdm(test_data):
        text = batch["text"]
        summary = batch["summary"]
        
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_length=128, 
                num_beams=4,
                no_repeat_ngram_size=2
            )
        
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(summary)

    # Compute Metrics
    scores = compute_metrics(predictions, references)
    print("\n" + "="*30)
    print("Final Test Set Results:")
    print(json.dumps(scores, indent=2))
    print("="*30)
    
    # Save results
    with open(os.path.join(model_dir, "test_results.json"), "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.path.join("results", "summarizer_mt5_small"))
    parser.add_argument("--subset", type=int, default=200, help="Number of test samples to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.subset)
