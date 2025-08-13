# features.py
import numpy as np
import pandas as pd
from datetime import datetime

def feature_engineering(X):
    df = X.copy()
    
    # Amount difference and ratio
    df["amount_diff"] = df["amount_paid"] - df["fee_amount_due"]
    df["amount_ratio"] = np.where(df["fee_amount_due"] != 0, df["amount_paid"] / df["fee_amount_due"], 0)
    
    # Overpayment flag
    df["is_over_payment"] = (df["amount_paid"] > df["fee_amount_due"]).astype(int)
    
    # Ensure binary columns are integers
    df["is_new_device"] = df["is_new_device"].astype(int)
    df["student_name_match"] = df["student_name_match"].astype(int)
    
    # Log time since last payment
    df["log_time_since_last_payment_days"] = np.log1p(df["time_since_last_payment_days"])
    
    # Parse timestamp and create cyclic features
    df["timestamp"] = df["timestamp"].apply(
        lambda ts: datetime.fromisoformat(ts.replace("Z", "+00:00")) if isinstance(ts, str) else ts
    )
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_day"] = np.sin(2 * np.pi * df["day"] / 7)
    df["cos_day"] = np.cos(2 * np.pi * df["day"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Select final features
    return df[[
        "amount_diff", "payment_method", "is_new_device", "student_name_match",
        "log_time_since_last_payment_days", "sin_hour", "cos_hour",
        "sin_day", "cos_day", "sin_month", "cos_month", "amount_ratio", "is_over_payment"
    ]]
