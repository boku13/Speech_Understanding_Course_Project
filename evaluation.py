import sys
import os
import numpy as np


def compute_det_curve(target_scores, nontarget_scores):
    """
    Compute detection error tradeoff (DET) curve
    
    Args:
        target_scores: Scores for truthful samples
        nontarget_scores: Scores for deceptive samples
        
    Returns:
        frr: False rejection rates
        far: False acceptance rates
        thresholds: Score thresholds
    """
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


def compute_eer(target_scores, nontarget_scores):
    """
    Returns equal error rate (EER) and the corresponding threshold.
    
    Args:
        target_scores: Scores for truthful samples
        nontarget_scores: Scores for deceptive samples
        
    Returns:
        eer: Equal error rate
        threshold: Threshold at EER
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def evaluate_eer(score_file, output_file=None):
    """
    Compute EER for lie detection system
    
    Args:
        score_file: Path to score file with format "file_id label score"
        output_file: Path to save evaluation results
        
    Returns:
        eer: Equal error rate as percentage
    """
    # Load scores
    data = np.genfromtxt(score_file, dtype=str)
    keys = data[:, 1]  # Labels (Deceptive/Truthful)
    scores = data[:, 2].astype(np.float64)  # Prediction scores
    
    # Extract truthful and deceptive scores
    # Higher score should indicate stronger support for the truthful class
    truthful_scores = scores[keys == 'Truthful']
    deceptive_scores = scores[keys == 'Deceptive']
    
    # Compute EER
    eer, threshold = compute_eer(truthful_scores, deceptive_scores)
    eer_percent = eer * 100
    
    # Print and save results
    if output_file:
        with open(output_file, "w") as f_res:
            f_res.write('\nLIE DETECTION SYSTEM\n')
            f_res.write('\tEER\t\t= {:8.4f} % (Equal error rate)\n'.format(eer_percent))
            f_res.write('\tThreshold\t= {:8.4f}\n'.format(threshold))
        os.system(f"cat {output_file}")
    
    return eer_percent
