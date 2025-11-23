# run_demo.py
import os
from src.data_loading import load_transactions, encode_addresses
from src.feature import make_features
from src.training import main as train_model
from src.detect import detect
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'data/second_order_df.csv'
MODEL_DIR = 'models'
REPORT_DIR = 'reports'

def run_pipeline(top_n=10):
    # 1. Load and preprocess
    print("Loading and preprocessing data...")
    df = load_transactions(DATA_PATH)
    df, _ = encode_addresses(df)

    # 2. Feature engineering
    print("Creating features...")
    feats = make_features(df)

    # 3. Train model (optional, skip if already trained)
    if not os.path.exists(f"{MODEL_DIR}/isoforest.joblib"):
        print("Training IsolationForest model...")
        train_model()  # you can adjust this to take DATA_PATH & MODEL_DIR if needed
    else:
        print("Model already trained, skipping training.")

    # 4. Run anomaly detection
    print("Running anomaly detection...")
    full, top = detect(DATA_PATH, top_n=top_n)

    # 5. Save top suspicious transactions
    os.makedirs(REPORT_DIR, exist_ok=True)
    top_csv_path = os.path.join(REPORT_DIR, 'top_suspicious_transactions.csv')
    top.to_csv(top_csv_path, index=False)
    print(f"Top {top_n} suspicious transactions saved to {top_csv_path}\n")

    # 6. Print top transactions
    print(top[['TxHash', 'anomaly_score', 'risk_score']])

    # 7. Plot histogram
    print("\nGenerating Risk Score histogram...")
    plt.hist(top['risk_score'], bins=20)
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    plt.title('Distribution of Risk Scores')
    plt.show()


if __name__ == "__main__":
    run_pipeline(top_n=10)
