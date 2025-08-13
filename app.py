from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
from pydantic import BaseModel
from datetime import datetime, timezone

# Load model, pipeline, and threshold
model = tf.keras.models.load_model("autoencoder_model.keras")
print("model loaded succesfully")
pipeline = joblib.load("main_pipeline.pkl")
print("pipeline loaded succesfully")
with open("threshold.json", "r") as f:
    threshold_data = json.load(f)
threshold = threshold_data["threshold"]
print("threshold loaded succesfully")

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
    print("✅ Endpoint triggered")
    try:
        print("✅ Step 1: Received request")
        # Convert Pydantic model to DataFrame
        df = pd.DataFrame([transaction.dict()])
        print("✅ Step 2: DataFrame created:", df.shape)

        print("✅ Step 3a: Starting pipeline transform")
        processed_txn = pipeline.transform(df)
        print("✅ Step 3b: Pipeline transformed data:", processed_txn.shape)

        print("✅ Step 4a: Starting model prediction")
        reconstructed = model.predict(processed_txn)
        print("✅ Step 4b: Model returned output:", reconstructed.shape)

        print("✅ Step 5a: Calculating reconstruction error")
        error = np.mean(np.square(processed_txn - reconstructed))
        print("✅ Step 5b: Reconstruction error calculated:", error)

        print("✅ Step 6a: Classifying anomaly")
        scale = classify_anomaly(error, threshold)
        print("✅ Step 6b: Classified anomaly as:", scale)

        return JSONResponse(content={
            "reconstruction_error": float(error),
            "threshold": threshold,
            "anomaly_scale": scale
        })
    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))