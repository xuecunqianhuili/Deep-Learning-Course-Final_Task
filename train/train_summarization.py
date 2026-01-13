import argparse
import os
import sys
import torch
from dataclasses import asdict

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from utils.metrics import compute_rouge
from data.load_datasets import load_summarization_dataset, tokenize_summarization
from model.summarizer import SummarizerConfig, get_model_and_tokenizer
from utils.visualize import plot_training_history, plot_data_distribution


def parse_args():
    parser = argparse.ArgumentParser(description="Train a summarization model (mT5 small)")
    parser.add_argument("--dataset", type=str, default="xlsum_zh", choices=["cnn_dailymail", "xsum", "xlsum_zh"], help="Dataset name")
    parser.add_argument("--subset_size", type=int, default=2000, help="Limit samples per split for quick training")
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="HF model name")
    parser.add_argument("--output_dir", type=str, default=os.path.join("results", "summarizer_mt5_small"))

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Training started with args:", args)
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
        
    os.makedirs(args.output_dir, exist_ok=True)


    config = SummarizerConfig(model_name=args.model_name)
    model, tokenizer = get_model_and_tokenizer(config)

    ds = load_summarization_dataset(args.dataset, subset_size=args.subset_size)
    
    # Visualize data distribution
    print("Visualizing data distribution...")
    plot_data_distribution(ds["train"]["text"], ds["train"]["summary"], args.output_dir)
    
    tokenized = tokenize_summarization(ds, tokenizer, args.max_input_length, args.max_target_length)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = preds.argmax(-1) if preds.ndim == 2 else preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 with pad_token_id
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_rouge(decoded_preds, decoded_labels)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        # mT5 is unstable with FP16. Use BF16 since RTX 5070 supports it perfectly.
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", tokenized["test"]),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()  # Ensure trainer_state.json is saved for plotting
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_history(args.output_dir)
    
    metrics = trainer.evaluate()

    # Save metrics to results
    import json
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    trainer.save_model(args.output_dir)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"summarizer_config": asdict(config), "train_args": vars(args)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
