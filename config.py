# src/config.py

feature_list = [
    "amount",          # corresponds to CSV Value
    "log_amount",      # log transform of amount
    "block_height",    # BlockHeight column
    "from_addr_hash",  # hashed From address
    "to_addr_hash",    # hashed To address
    "hour",            # hour of transaction from TimeStamp
    "dayofweek",       # day of week from TimeStamp
    "is_error"         # 0/1 if transaction failed
]


