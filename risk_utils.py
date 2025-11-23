# src/risk_utils.py
import numpy as np

def score_to_risk(scores):
    """
    Convert IsolationForest score_samples (higher = more normal)
    to Risk Score 0-100 (higher = more risky).
    """
    scores = np.array(scores)
    inv = -scores  # invert so higher = more anomalous
    ranks = np.argsort(np.argsort(inv))  # 0..n-1
    percentiles = ranks / (len(inv) - 1)
    risk_scores = (percentiles * 100).round(2)
    return risk_scores
