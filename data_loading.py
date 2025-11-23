# src/data_loading.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_transactions(path, nrows=None):
    """
    Load blockchain transactions from CSV and normalize column names
    for the rest of the pipeline.
    Expects columns like:
    Unnamed: 0, TxHash, BlockHeight, TimeStamp, From, To, Value, isError
    """
    df = pd.read_csv(path, nrows=nrows)

    # Drop useless index column if present
    for col in ['Unnamed: 0', 'Unnamed: 0.1']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Normalize column names to what the rest of the code expects
    # Map your real columns to generic ones
    if 'TxHash' in df.columns:
        df['tx_hash'] = df['TxHash']
    if 'From' in df.columns:
        df['from_addr'] = df['From']
    if 'To' in df.columns:
        df['to_addr'] = df['To']
    if 'Value' in df.columns:
        df['amount'] = df['Value']
    if 'TimeStamp' in df.columns:
        df['timestamp'] = df['TimeStamp']

    # Drop rows with missing amount
    df = df.dropna(subset=['amount'])

    # Ensure amount is numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])

    # Parse timestamp
    try:
        # try epoch seconds
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    except Exception:
        # generic parse
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    return df

def encode_addresses(df, cols=['from_addr', 'to_addr']):
    """
    Label-encode address columns into numeric features.
    """
    encoders = {}
    for c in cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c + '_enc'] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
    return df, encoders

