from fastapi import FastAPI
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

# --- 1. PRE-INITIALIZE VARIABLES ---
# This prevents the "NameError: model is not defined" if the load fails
model = None
scaler = None
pca = None
le = None

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 2. MODEL LOADING ---
try:
    model_path = os.path.join(BASE_DIR, 'ids_model.h5')
    # Use compile=False to bypass version-specific training configurations
    model = tf.keras.models.load_model(model_path, compile=False)
    # Re-compile manually for inference
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 3. ASSET LOADING ---
try:
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
    pca = joblib.load(os.path.join(BASE_DIR, 'pca_model.joblib'))
    le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.joblib'))
    print("All assets (scaler, pca, le) loaded successfully!")
except Exception as e:
    print(f"Error loading assets: {e}")

@app.get("/")
def home():
    # Now this will return model_loaded: False instead of crashing if loading fails
    return {
        "status": "IDS API is running", 
        "model_loaded": model is not None,
        "assets_loaded": all([scaler, pca, le])
    }

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded on server. Check logs for TensorFlow version errors."}
    
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
        return {"error": f"Prediction failed: {str(e)}"}
