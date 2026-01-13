import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualize import plot_training_history

output_dir = os.path.join("results", "summarizer_mt5_small")
print(f"Regenerating plots from {output_dir}...")
plot_training_history(output_dir)
