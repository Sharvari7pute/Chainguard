# src/training.py
import argparse
import os
import joblib

from src.config import feature_list

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_transactions, encode_addresses
from src.feature import make_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Path to CSV (e.g. data/second_order_df.csv)")
    parser.add_argument('--out', default='models', help="Output directory for model files")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('Loading data...')
    df = load_transactions(args.data)

    print('Encoding addresses...')
    df, encs = encode_addresses(df)

    print('Making features...')
    X = make_features(df)

    print('Scaling...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print('Training IsolationForest...')
    clf = IsolationForest(
        n_estimators=200,
        contamination=0.01,  # adjust based on how many anomalies you expect
        random_state=42
    )
    clf.fit(X_scaled)

    joblib.dump(clf, os.path.join(args.out, 'isoforest.joblib'))
    joblib.dump(scaler, os.path.join(args.out, 'scaler.joblib'))
    joblib.dump(encs, os.path.join(args.out, 'encoders.joblib'))

    print('Saved models to', args.out)

if __name__ == '__main__':
    main()

