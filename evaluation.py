import sys
import os
import numpy as np
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def calculate_lie_detection_metrics(eval_score_path, output_file=None):
    """
    Calculate comprehensive metrics for lie detection including EER and ROC AUC.
    
    Args:
        eval_score_path: Path to the evaluation scores file
        output_file: Optional path to write detailed results
        
    Returns:
        Dictionary containing metrics including EER, AUC, and other performance metrics
    """
    # Check if file exists
    if not os.path.exists(eval_score_path):
        raise FileNotFoundError(f"Score file not found: {eval_score_path}")
    # print("HEELLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO",eval_score_path)
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
    
    # Count truthful and deceptive samples
    truthful_count = np.sum(y_true == 1)
    deceptive_count = np.sum(y_true == 0)
    
    # Count predictions
    truthful_pred_count = np.sum(y_pred == 1)
    deceptive_pred_count = np.sum(y_pred == 0)
    
    # Calculate standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER
    # Separate scores for truthful and deceptive samples
    truthful_scores = y_scores[y_true == 1]
    deceptive_scores = y_scores[y_true == 0]
    # print("heheheheheheheheheheheheheheheheheheheheheheheheh")
    #This allows the EER calculation to determine at what threshold the classification errors balance out between the two classes.
    
    if len(truthful_scores) > 0 and len(deceptive_scores) > 0:
        eer, eer_threshold = compute_eer_fixed(truthful_scores, deceptive_scores)
    else:
        eer, eer_threshold = 0.5, 0.5  # Default if no samples of one class
    
    # Compile all metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": roc_auc,
        "truthful_count": int(truthful_count),
        "deceptive_count": int(deceptive_count),
        "truthful_pred_count": int(truthful_pred_count),
        "deceptive_pred_count": int(deceptive_pred_count)
    }
    
    # Write detailed results to file if requested
    if output_file:
        with open(output_file, "w") as f:
            f.write("\nLIE DETECTION METRICS\n")
            f.write(f"\tAccuracy\t= {accuracy*100:8.3f}%\n")
            f.write(f"\tPrecision\t= {precision*100:8.3f}%\n")
            f.write(f"\tRecall\t\t= {recall*100:8.3f}%\n")
            f.write(f"\tF1 Score\t= {f1*100:8.3f}%\n")
            f.write(f"\tEER\t\t= {eer*100:8.3f}%\n")
            f.write(f"\tROC AUC\t\t= {roc_auc:8.3f}\n")
            f.write(f"\tEER Threshold\t= {eer_threshold:8.3f}\n")
            
            # Add class distribution information
            f.write("\nCLASS DISTRIBUTION\n")
            f.write(f"\tTruthful samples\t= {truthful_count} ({truthful_count/(truthful_count+deceptive_count)*100:.1f}%)\n")
            f.write(f"\tDeceptive samples\t= {deceptive_count} ({deceptive_count/(truthful_count+deceptive_count)*100:.1f}%)\n")
            f.write(f"\tTruthful predictions\t= {truthful_pred_count} ({truthful_pred_count/(truthful_pred_count+deceptive_pred_count)*100:.1f}%)\n")
            f.write(f"\tDeceptive predictions\t= {deceptive_pred_count} ({deceptive_pred_count/(truthful_pred_count+deceptive_pred_count)*100:.1f}%)\n")
            
            # Add confusion matrix information
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            false_positives = np.sum((y_true == 0) & (y_pred == 1))
            true_negatives = np.sum((y_true == 0) & (y_pred == 0))
            false_negatives = np.sum((y_true == 1) & (y_pred == 0))
            
            f.write("\nCONFUSION MATRIX\n")
            f.write(f"\tTrue Positives (Truthful correctly identified)\t= {true_positives}\n")
            f.write(f"\tFalse Positives (Deceptive misclassified as Truthful)\t= {false_positives}\n")
            f.write(f"\tTrue Negatives (Deceptive correctly identified)\t= {true_negatives}\n")
            f.write(f"\tFalse Negatives (Truthful misclassified as Deceptive)\t= {false_negatives}\n")
            
            # Add score distribution information
            f.write("\nSCORE DISTRIBUTION\n")
            f.write(f"\tMin score\t= {min(scores):.4f}\n")
            f.write(f"\tMax score\t= {max(scores):.4f}\n")
            f.write(f"\tMean score\t= {np.mean(scores):.4f}\n")
            f.write(f"\tStd score\t= {np.std(scores):.4f}\n")
            
        print(f"Detailed metrics saved to {output_file}")
    
    return metrics



def plot_performance_curves(eval_score_path, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    import seaborn as sns
    import os
    
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
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    
    # Calculate and plot FPR and FNR with EER point
    thresholds_roc = np.append(thresholds, 1.0)  # Add a final threshold
    fnr = 1 - tpr  # False Negative Rate = 1 - TPR
    
    # Find EER point (where FPR = FNR)
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds_roc[eer_index]
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds_roc, fpr, color='red', lw=2, label='False Positive Rate (FPR)')
    plt.plot(thresholds_roc, fnr, color='blue', lw=2, label='False Negative Rate (FNR)')
    plt.scatter([eer_threshold], [eer], color='purple', s=100, label=f'EER = {eer:.3f} at threshold = {eer_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('FPR-FNR Curves with Equal Error Rate (EER) Point')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'eer_curve.png'), dpi=300)
    
    # Generate histograms of scores for truthful and deceptive classes
    plt.figure(figsize=(10, 8))
    bins = np.linspace(0, 1, 50)
    plt.hist(truthful_scores, bins=bins, alpha=0.5, label='Truthful', color='green')
    plt.hist(deceptive_scores, bins=bins, alpha=0.5, label='Deceptive', color='red')
    plt.axvline(x=eer_threshold, color='black', linestyle='--', label=f'EER Threshold = {eer_threshold:.3f}')
    plt.xlabel('Score (Truthful Probability)')
    plt.ylabel('Count')
    plt.title('Distribution of Scores by Class')
    plt.legend(loc="upper center")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300)
    
    # NEW: Create Confusion Confidence Matrix
    # Define confidence bins
    confidence_bins = 5  # Number of confidence level divisions
    bin_edges = np.linspace(0, 1, confidence_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # We need to transform scores so they represent confidence in the predicted class
    # For predictions of "Truthful" (y_pred=1), the confidence is the score
    # For predictions of "Deceptive" (y_pred=0), the confidence is 1-score
    confidence_scores = np.where(y_pred == 1, y_scores, 1 - y_scores)
    
    # Initialize confusion confidence matrix
    # Rows are true classes (0: Deceptive, 1: Truthful)
    # Columns are predicted classes and confidence levels
    confusion_confidence = np.zeros((2, 2*confidence_bins))
    
    # Fill the confusion confidence matrix
    for true_class in [0, 1]:  # 0: Deceptive, 1: Truthful
        for pred_class in [0, 1]:  # 0: Deceptive, 1: Truthful
            mask = (y_true == true_class) & (y_pred == pred_class)
            if np.sum(mask) > 0:
                # Get confidence scores for this combination
                these_confidences = confidence_scores[mask]
                # Bin the confidences
                hist, _ = np.histogram(these_confidences, bins=bin_edges)
                # Fill the appropriate slice of the matrix
                start_col = pred_class * confidence_bins
                end_col = (pred_class + 1) * confidence_bins
                confusion_confidence[true_class, start_col:end_col] = hist
    
    # Normalize confusion confidence matrix by row (true class)
    row_sums = confusion_confidence.sum(axis=1, keepdims=True)
    confusion_confidence_norm = np.zeros_like(confusion_confidence, dtype=float)
    for i in range(confusion_confidence.shape[0]):
        if row_sums[i] > 0:
            confusion_confidence_norm[i] = confusion_confidence[i] / row_sums[i]
    
    # Create labels for the heatmap
    confidence_labels = []
    for pred_class in ["Deceptive", "Truthful"]:
        for conf_level in range(confidence_bins):
            lower = bin_edges[conf_level]
            upper = bin_edges[conf_level + 1]
            confidence_labels.append(f"{pred_class}\n{lower:.1f}-{upper:.1f}")
    
    true_class_labels = ["Deceptive", "Truthful"]
    
    # Plot the confusion confidence matrix
    plt.figure(figsize=(14, 8))
    sns.heatmap(confusion_confidence_norm, annot=confusion_confidence, fmt='g', cmap='viridis',
                xticklabels=confidence_labels, yticklabels=true_class_labels, cbar_kws={'label': 'Proportion of True Class'})
    plt.xlabel('Predicted Class and Confidence Level')
    plt.ylabel('True Class')
    plt.title('Confusion Confidence Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_confidence_matrix.png'), dpi=300)
    
    # Create a simpler, more readable version with fewer bins
    # This version uses just 3 confidence levels: low, medium, high
    simple_confidence_bins = 3
    simple_bin_edges = np.linspace(0, 1, simple_confidence_bins+1)
    simple_bin_names = ["Low", "Medium", "High"]
    
    # Initialize simpler confusion confidence matrix
    simple_conf_conf = np.zeros((2, 2*simple_confidence_bins))
    
    # Fill the simpler confusion confidence matrix
    for true_class in [0, 1]:  # 0: Deceptive, 1: Truthful
        for pred_class in [0, 1]:  # 0: Deceptive, 1: Truthful
            mask = (y_true == true_class) & (y_pred == pred_class)
            if np.sum(mask) > 0:
                # Get confidence scores for this combination
                these_confidences = confidence_scores[mask]
                # Bin the confidences
                hist, _ = np.histogram(these_confidences, bins=simple_bin_edges)
                # Fill the appropriate slice of the matrix
                start_col = pred_class * simple_confidence_bins
                end_col = (pred_class + 1) * simple_confidence_bins
                simple_conf_conf[true_class, start_col:end_col] = hist
    
    # Normalize by row (true class)
    simple_row_sums = simple_conf_conf.sum(axis=1, keepdims=True)
    simple_conf_conf_norm = np.zeros_like(simple_conf_conf, dtype=float)
    for i in range(simple_conf_conf.shape[0]):
        if simple_row_sums[i] > 0:
            simple_conf_conf_norm[i] = simple_conf_conf[i] / simple_row_sums[i]
    
    # Create labels for the simpler heatmap
    simple_confidence_labels = []
    for pred_class in ["Deceptive", "Truthful"]:
        for conf_level in simple_bin_names:
            simple_confidence_labels.append(f"{pred_class}\n{conf_level}")
    
    # Plot the simpler confusion confidence matrix
    plt.figure(figsize=(14, 8))
    sns.heatmap(simple_conf_conf_norm, annot=simple_conf_conf, fmt='g', cmap='viridis',
                xticklabels=simple_confidence_labels, yticklabels=true_class_labels, cbar_kws={'label': 'Proportion of True Class'})
    plt.xlabel('Predicted Class and Confidence Level')
    plt.ylabel('True Class')
    plt.title('Confusion Confidence Matrix (Simplified)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_confidence_matrix_simple.png'), dpi=300)
    
    # Summary results
    results = {
        'auc': roc_auc,
        'average_precision': avg_precision,
        'eer': eer,
        'eer_threshold': eer_threshold
    }
    
    return results


def calculate_tDCF_EER(cm_scores_file,
                       asv_score_file,
                       output_file,
                       printout=True):
    # Replace CM scores with your own scores or provide score file as the
    # first argument.
    # cm_scores_file =  'score_cm.txt'
    # Replace ASV scores with organizers' scores or provide score file as
    # the second argument.
    # asv_score_file = 'ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv':
        10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    # asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    # cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to
    # EER threshold
    eer_asv, asv_threshold = compute_eer_fixed(tar_asv, non_asv)
    eer_cm = compute_eer_fixed(bona_cm, spoof_cm)[0]

    attack_types = [f'A{_id:02d}' for _id in range(7, 20)]
    if printout:
        spoof_cm_breakdown = {
            attack_type: cm_scores[cm_sources == attack_type]
            for attack_type in attack_types
        }

        eer_cm_breakdown = {
            attack_type: compute_eer_fixed(bona_cm,
                                     spoof_cm_breakdown[attack_type])[0]
            for attack_type in attack_types
        }

    [Pfa_asv, Pmiss_asv,
     Pmiss_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                               asv_threshold)

    # Compute t-DCF
    tDCF_curve, CM_thresholds = compute_tDCF(bona_cm,
                                             spoof_cm,
                                             Pfa_asv,
                                             Pmiss_asv,
                                             Pmiss_spoof_asv,
                                             cost_model,
                                             print_cost=False)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(Equal error rate for countermeasure)\n'.format(
                            eer_cm * 100))

            f_res.write('\nTANDEM\n')
            f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))

            f_res.write('\nBREAKDOWN CM SYSTEM\n')
            for attack_type in attack_types:
                _eer = eer_cm_breakdown[attack_type] * 100
                f_res.write(
                    f'\tEER {attack_type}\t\t= {_eer:8.9f} % (Equal error rate for {attack_type}\n'
                )
        os.system(f"cat {output_file}")

    return eer_cm * 100, min_tDCF


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer_fixed(target_scores, nontarget_scores):
    """Compute EER using scikit-learn's ROC curve implementation"""
    # Create binary labels (1 for target/truthful, 0 for nontarget/deceptive)
    y_true = np.concatenate([np.ones(len(target_scores)), np.zeros(len(nontarget_scores))])
    # Concatenate all scores (keeping target scores first, then nontarget scores)
    y_scores = np.concatenate([target_scores, nontarget_scores])
    
    # Compute ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Print values to help with debugging
    print("FPR:", fpr)
    print("FNR:", fnr)
    print("Thresholds:", thresholds)

    # Handle the case where there might not be a point where FPR and FNR cross
    if np.min(np.abs(fpr - fnr)) > 0.1:  # Significant gap between curves
        print("Warning: FPR and FNR curves don't cross smoothly. EER may be approximate.")
        
        # If we have few points, try to interpolate
        if len(fpr) < 10:
            print("Few threshold points available. Consider using more data.")
    
    # Find the threshold where FPR and FNR are closest
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    
    # Add more informative output
    print(f"EER calculation point: FPR={fpr[eer_threshold_idx]:.4f}, FNR={fnr[eer_threshold_idx]:.4f}")
    print(f"Calculated EER: {eer:.4f} at threshold: {thresholds[eer_threshold_idx]:.4f}")
    
    return eer, thresholds[eer_threshold_idx]
def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
    """

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit(
            'ERROR: Your prior probabilities should be positive and sum up to one.'
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            'ERROR: you should provide miss rate of spoof tests against your ASV system.'
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit(
            'ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
        cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?'
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(
            bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.
              format(cost_model['Ptar']))
        print(
            '   Pnon         = {:8.5f} (Prior probability of nontarget user)'.
            format(cost_model['Pnon']))
        print(
            '   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.
            format(cost_model['Pspoof']))
        print(
            '   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'
            .format(cost_model['Cfa_asv']))
        print(
            '   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'
            .format(cost_model['Cmiss_asv']))
        print(
            '   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'
            .format(cost_model['Cfa_cm']))
        print(
            '   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'
            .format(cost_model['Cmiss_cm']))
        print(
            '\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)'
        )

        if C2 == np.minimum(C1, C2):
            print(
                '   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(
                    C1 / C2))
        else:
            print(
                '   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(
                    C2 / C1))

    return tDCF_norm, CM_thresholds