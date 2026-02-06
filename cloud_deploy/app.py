from fastapi import FastAPI
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

# --- Model Loading with Version Compatibility ---
try:
    # compile=False allows us to load the model even if the Keras versions differ
    model = tf.keras.models.load_model('ids_model.h5', compile=False)
    # Manually compile with basic settings since we only need it for prediction
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the other assets
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca_model.joblib')
le = joblib.load('label_encoder.joblib')

@app.get("/")
def home():
    return {"status": "IDS API is running", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert incoming JSON data to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocessing (Z-Score Normalization)
        scaled_data = scaler.transform(df)
        
        # Feature Engineering (PCA)
        pca_data = pca.transform(scaled_data)
        
        # Prediction (DNN Inference)
        prediction = model.predict(pca_data)
        class_idx = np.argmax(prediction)
        
        return {
            "result": str(le.classes_[class_idx]),
            "confidence": float(np.max(prediction))
        }
    except Exception as e:
        return {"error": str(e)}