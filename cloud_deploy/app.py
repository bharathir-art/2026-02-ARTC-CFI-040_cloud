import os
# This tells TensorFlow to use the legacy bridge we just added to requirements
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
import tf_keras as keras # Use the bridge directly

app = FastAPI()

model = None
scaler = None
pca = None
le = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ASSET LOADING ---
try:
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
    pca = joblib.load(os.path.join(BASE_DIR, 'pca_model.joblib'))
    le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.joblib'))
    print("Assets loaded successfully.")
except Exception as e:
    print(f"Asset Error: {e}")

# --- MODEL LOADING ---
try:
    model_path = os.path.join(BASE_DIR, 'ids_model.h5')
    # Use the legacy keras bridge to load the older .h5 format
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print("Model loaded successfully using legacy bridge!")
except Exception as e:
    print(f"Model load failed: {e}")

@app.get("/")
def home():
    return {
        "status": "IDS API is running", 
        "model_loaded": model is not None,
        "assets_loaded": all([scaler, pca, le])
    }
