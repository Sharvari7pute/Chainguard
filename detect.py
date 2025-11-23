# src/detect.py
from src.config import feature_list
import joblib
import pandas as pd
import os

from src.data_loading import load_transactions, encode_addresses
from src.feature import make_features
from src.risk_utils import score_to_risk

MODEL_DIR = 'models'

def detect(path, top_n=100):
    """
    Run anomaly detection on a CSV file and return a DataFrame
    with anomaly scores, binary labels, and risk scores.
    """
    # 1. Load and preprocess
    df = load_transactions(path)
    df, encs = encode_addresses(df)
    print('Making features...')
    X = make_features(df)

    # 2. Load trained model and scaler
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    clf = joblib.load(f"{MODEL_DIR}/isoforest.joblib")

    # 3. Scale features and get scores
    Xs = scaler.transform(X)

    # decision_function: higher = more normal, lower = more anomalous
    scores = clf.decision_function(Xs)
    labels = clf.predict(Xs)  # -1 anomaly, 1 normal

    # 4. Convert to anomaly_score and risk_score
    df['anomaly_score'] = -scores  # higher = more anomalous
    df['is_anomaly'] = (labels == -1).astype(int)

    # risk_score 0–100 derived from anomaly_score
    s = df['anomaly_score']
    if s.max() > s.min():
        df['risk_score'] = ((s - s.min()) / (s.max() - s.min()) * 100).round(2)
    else:
        df['risk_score'] = 0.0

    # Sort by risk_score descending
    df_sorted = df.sort_values('risk_score', ascending=False)

    # Deduplicate by TxHash so each transaction appears only once
    df_sorted = df_sorted.drop_duplicates(subset='TxHash')

    # Keep top_n for report
    top = df_sorted.head(top_n).copy()

    return df_sorted, top

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.detect <path-to-csv>")
        raise SystemExit(1)

    # Run detection
    full, top = detect(sys.argv[1], top_n=20)

    # Ensure reports folder exists
    os.makedirs('reports', exist_ok=True)

    # Save top 20 suspicious transactions to CSV
    top.to_csv('reports/top_suspicious_transactions.csv', index=False)

    # Print a compact view of top 20
    cols_to_show = []
    for c in ['TxHash', 'Value', 'anomaly_score', 'risk_score', 'is_anomaly']:
        if c in top.columns:
            cols_to_show.append(c)

    print(top[cols_to_show].to_string(index=False))
