# Plot Generator for Lie Detection Results

This script generates performance plots for lie detection results using the evaluation scores file from the AASIST framework.

## Prerequisites

Make sure you have the following Python packages installed:

```bash
pip install numpy matplotlib scikit-learn seaborn
```

## Usage

Run the script with the path to the evaluation scores file:

```bash
python generate_plots.py --scores Rodecar_Results/LieDetection_AASIST_ep50_bs32/eval_scores_using_best_dev_model.txt --output performance_plots
```

## Arguments

- `--scores`: Path to the evaluation scores file (required)
- `--output`: Directory where plots will be saved (default: "performance_plots")

## Generated Plots

The script generates the following plots:

1. **ROC Curve** (`roc_curve.png`): The Receiver Operating Characteristic curve showing the tradeoff between true positive rate and false positive rate.

2. **Precision-Recall Curve** (`precision_recall_curve.png`): Shows the tradeoff between precision and recall.

3. **ROC Curve with EER** (`roc_curve_with_eer.png`): ROC curve with the Equal Error Rate point marked.

4. **Score Distributions** (`score_distributions.png`): Histograms of truthful and deceptive scores with the EER threshold marked.

5. **Confusion Confidence Matrix** (`confusion_confidence_matrix.png`): A heatmap showing the distribution of predictions across ground truth values.

## Output Metrics

The script also outputs key performance metrics:
- ROC AUC (Area Under the ROC Curve)
- EER (Equal Error Rate)
- EER Threshold (the decision threshold at which the EER is achieved)

## Example

```bash
$ python generate_plots.py --scores Rodecar_Results/LieDetection_AASIST_ep50_bs32/eval_scores_using_best_dev_model.txt --output lie_detection_plots

Generating plots from Rodecar_Results/LieDetection_AASIST_ep50_bs32/eval_scores_using_best_dev_model.txt
EER calculation point: FPR=0.1847, FNR=0.1857
Calculated EER: 0.1852 at threshold: 0.5706

==================================================
RESULTS:
==================================================
ROC AUC:      0.886
EER:          18.520%
EER Threshold: 0.5706
==================================================
Plots saved to lie_detection_plots
```

## Note

The script automatically extracts ground truth labels from filenames based on keywords like "truth", "truthful", "lie", and "deceptive". If the filename format is different, you may need to adjust the script accordingly. 