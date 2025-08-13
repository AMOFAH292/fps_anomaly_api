from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
from pydantic import BaseModel
from datetime import datetime

# Load model, scaler, encoder, and threshold
model = tf.keras.models.load_model("autoencoder_model.keras")
print("model loaded successfully")

scaler = joblib.load("scaler.pkl")
print("scaler loaded successfully")

encoder = joblib.load("encoder.pkl")
print("encoder loaded successfully")

with open("threshold.json", "r") as f:
    threshold_data = json.load(f)
threshold = threshold_data["threshold"]
print("threshold loaded successfully")

# Import feature engineering
from features import feature_engineering

# Pydantic model for request validation
class Transaction(BaseModel):
    fee_amount_due: float
    amount_paid: float
    payment_method: str
    student_type: str
    is_new_device: bool
    student_name_match: bool
    time_since_last_payment_days: int
    timestamp: str

# Helper function
def classify_anomaly(error, threshold):
    if error < threshold:
        return "Low"
    elif error < threshold * 10:
        return "Medium"
    else:
        return "High"

# FastAPI app
app = FastAPI(title="Anomaly Detection API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Anomaly Detection API is running"}

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Convert Pydantic model to DataFrame
        df = pd.DataFrame([transaction.dict()])

        # Apply feature engineering
        df_fe = feature_engineering(df)

        # Scale numeric columns
        numeric_cols = ['amount_diff', 'log_time_since_last_payment_days', 'sin_hour', 'cos_hour',
                        'sin_day', 'cos_day', 'sin_month', 'cos_month', 'amount_ratio']
        df_fe[numeric_cols] = scaler.transform(df_fe[numeric_cols])

        # Encode categorical column
        encoded_payment = encoder.transform(df_fe[['payment_method']])
        encoded_cols = encoder.get_feature_names_out(['payment_method'])
        df_encoded = pd.DataFrame(encoded_payment, columns=encoded_cols, index=df_fe.index)

        # Combine all features for model
        binary_cols = ['is_new_device', 'student_name_match', 'is_over_payment']
        model_input = pd.concat([df_fe[numeric_cols + binary_cols], df_encoded], axis=1)

        # Make prediction
        reconstructed = model.predict(model_input)
        error = np.mean(np.square(model_input - reconstructed))
        scale = classify_anomaly(error, threshold)

        return JSONResponse(content={
            "reconstruction_error": float(error),
            "threshold": threshold,
            "anomaly_scale": scale
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
