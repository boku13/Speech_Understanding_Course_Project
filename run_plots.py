import os
from pathlib import Path
from main import plot_performance_curves

# Set the exact paths as used in the original main.py
eval_score_path = Path("Rodecar_Results/LieDetection_AASIST_ep50_bs32/eval_scores_using_best_dev_model.txt")
output_dir = Path("Rodecar_Results/LieDetection_AASIST_ep50_bs32/performance_plots")

# Make sure the paths exist
if not eval_score_path.exists():
    raise FileNotFoundError(f"Evaluation scores file not found: {eval_score_path}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("Generating performance plots with Confusion Confidence Matrix...")
plot_results = plot_performance_curves(
    eval_score_path,
    output_dir
)

print(f"Performance plots saved to {output_dir}")
print(f"ROC AUC: {plot_results['auc']:.3f}")
print(f"EER: {plot_results['eer']*100:.3f}% at threshold {plot_results['eer_threshold']:.3f}") 