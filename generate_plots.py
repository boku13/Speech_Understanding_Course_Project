import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import argparse
from pathlib import Path


def compute_eer_fixed(target_scores, nontarget_scores):
    """Compute EER using scikit-learn's ROC curve implementation"""
    # Create binary labels (1 for target/truthful, 0 for nontarget/deceptive)
    y_true = np.concatenate([np.ones(len(target_scores)), np.zeros(len(nontarget_scores))])
    # Concatenate all scores (keeping target scores first, then nontarget scores)
    y_scores = np.concatenate([target_scores, nontarget_scores])
    
    # Compute ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are closest
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    
    print(f"EER calculation point: FPR={fpr[eer_threshold_idx]:.4f}, FNR={fnr[eer_threshold_idx]:.4f}")
    print(f"Calculated EER: {eer:.4f} at threshold: {thresholds[eer_threshold_idx if eer_threshold_idx < len(thresholds) else -1]:.4f}")
    
    return eer, thresholds[eer_threshold_idx if eer_threshold_idx < len(thresholds) else -1]


def plot_performance_curves(eval_score_path, output_dir):
    """Generate performance plots from evaluation scores file"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read scores file
    with open(eval_score_path, "r") as f:
        lines = f.readlines()
    
    filenames = []
    predictions = []
    scores = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            filename = parts[0]
            prediction = parts[1]
            score = float(parts[2])
            
            filenames.append(filename)
            predictions.append(prediction)
            scores.append(score)
    
    # Extract ground truth from filenames
    ground_truth = []
    for filename in filenames:
        if "truth" in filename.lower() or "truthful" in filename.lower():
            ground_truth.append("Truthful")
        elif "lie" in filename.lower() or "deceptive" in filename.lower():
            ground_truth.append("Deceptive")
        else:
            print(f"Warning: Cannot determine ground truth from filename: {filename}")
    
    # Convert to binary for metrics calculation
    y_true = np.array([1 if label == "Truthful" else 0 for label in ground_truth])
    y_pred = np.array([1 if label == "Truthful" else 0 for label in predictions])
    y_scores = np.array(scores)
    
    # Separate scores for truthful and deceptive samples
    truthful_scores = y_scores[y_true == 1]
    deceptive_scores = y_scores[y_true == 0]
    
    # Calculate EER
    eer, eer_threshold = compute_eer_fixed(truthful_scores, deceptive_scores)
    
    # Set up Seaborn style
    sns.set(style="whitegrid")
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.fill_between(recall, precision, alpha=0.2, color='green')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()
    
    # Plot EER point on ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Find closest point to EER on the ROC curve
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10, 
             label=f'EER = {eer:.4f} @ threshold = {eer_threshold:.4f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Equal Error Rate (EER) Point')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve_with_eer.png'), dpi=300)
    plt.close()
    
    # Plot score distributions
    plt.figure(figsize=(12, 8))
    sns.histplot(truthful_scores, color='green', alpha=0.6, label='Truthful', bins=30, kde=True)
    sns.histplot(deceptive_scores, color='red', alpha=0.6, label='Deceptive', bins=30, kde=True)
    plt.axvline(x=eer_threshold, color='black', linestyle='--', 
                label=f'EER Threshold = {eer_threshold:.4f}')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distributions with EER Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=300)
    plt.close()
    
    # Plot Confusion Confidence Matrix
    confusion_scores = np.zeros((len(y_true), 2))
    for i, (truth, score) in enumerate(zip(y_true, y_scores)):
        confusion_scores[i, 0] = 1 - truth  # 0 if truthful, 1 if deceptive (ground truth)
        confusion_scores[i, 1] = 1 - score  # Confidence of being deceptive
    
    # Create bins for the heatmap
    bins = 10
    heatmap_data = np.zeros((bins, bins))
    
    # Count samples in each bin
    x_edges = np.linspace(0, 1, bins+1)
    y_edges = np.linspace(0, 1, bins+1)
    
    for sample in confusion_scores:
        x_bin = min(int(sample[0] * bins), bins-1)
        y_bin = min(int(sample[1] * bins), bins-1)
        heatmap_data[y_bin, x_bin] += 1
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", 
                    xticklabels=[f"{x:.1f}" for x in np.linspace(0, 1, bins)],
                    yticklabels=[f"{y:.1f}" for y in np.linspace(0, 1, bins)])
    
    plt.xlabel('Ground Truth (0=Truthful, 1=Deceptive)')
    plt.ylabel('Model Prediction Confidence (0=Truthful, 1=Deceptive)')
    plt.title('Confusion Confidence Matrix')
    
    # Add diagonal line to show perfect prediction
    plt.plot([0, bins], [0, bins], 'r--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_confidence_matrix.png'), dpi=300)
    plt.close()
    
    # Return results
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": roc_auc
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance plots from evaluation scores")
    parser.add_argument("--scores", type=str, required=True, 
                        help="Path to evaluation scores file")
    parser.add_argument("--output", type=str, default="performance_plots",
                        help="Directory to save plots (default: performance_plots)")
    
    args = parser.parse_args()
    
    eval_score_path = Path(args.scores)
    output_dir = Path(args.output)
    
    print(f"Generating plots from {eval_score_path}")
    results = plot_performance_curves(eval_score_path, output_dir)
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"ROC AUC:      {results['auc']:.3f}")
    print(f"EER:          {results['eer']*100:.3f}%")
    print(f"EER Threshold: {results['eer_threshold']:.4f}")
    print("="*50)
    print(f"Plots saved to {output_dir}") 