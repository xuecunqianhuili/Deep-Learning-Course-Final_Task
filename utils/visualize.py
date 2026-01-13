import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_training_history(output_dir: str):
    """Plot Loss and ROUGE scores from trainer_state.json."""
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"Warning: {state_path} not found.")
        return

    with open(state_path, "r") as f:
        state = json.load(f)

    epochs = []
    loss = []
    eval_rouge1 = []
    eval_rougeL = []

    for log in state["log_history"]:
        if "loss" in log and "epoch" in log:
            epochs.append(log["epoch"])
            loss.append(log["loss"])
        elif "eval_rouge1" in log:
            eval_rouge1.append(log["eval_rouge1"])
            eval_rougeL.append(log["eval_rougeL"])

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    # Plot ROUGE (Only if valid values exist)
    if eval_rouge1 and any(r > 0 for r in eval_rouge1):
        plt.subplot(1, 2, 2)
        plt.plot(range(len(eval_rouge1)), eval_rouge1, label="ROUGE-1")
        plt.plot(range(len(eval_rougeL)), eval_rougeL, label="ROUGE-L")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Score")
        plt.title("ROUGE Metrics")
        plt.legend()
    else:
        # If ROUGE is missing or 0, just plot Loss in a single nice chart
        plt.clf() # Clear previous
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, label="Training Loss", color='tab:blue', linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Loss Curve", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved training curves to {save_path}")


def plot_data_distribution(texts: list, summaries: list, save_dir: str):
    """Plot character count distribution (for Chinese texts)."""
    # Try to support Chinese display in matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    text_lens = [len(str(t)) for t in texts]
    summary_lens = [len(str(s)) for s in summaries]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(text_lens, bins=30, color='skyblue', edgecolor='black')
    plt.title("文章字数分布")
    plt.xlabel("字数")
    plt.ylabel("频率")

    plt.subplot(1, 2, 2)
    plt.hist(summary_lens, bins=30, color='salmon', edgecolor='black')
    plt.title("摘要字数分布")
    plt.xlabel("字数")
    plt.ylabel("频率")


    plt.tight_layout()
    save_path = os.path.join(save_dir, "data_distribution.png")
    plt.savefig(save_path)
    print(f"Saved data distribution to {save_path}")
