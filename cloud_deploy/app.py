import os
# Try to force Keras 2 behavior if Keras 3 fails
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

from fastapi import FastAPI
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

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
    print("Assets loaded.")
except Exception as e:
    print(f"Asset Error: {e}")

# --- MODEL LOADING (Dual-Strategy) ---
try:
    model_path = os.path.join(BASE_DIR, 'ids_model.h5')
    
    # Strategy A: Standard Load
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print("Strategy A Success: Model loaded.")
except Exception as e:
    print(f"Strategy A failed, trying Strategy B... Error: {e}")
    try:
        # Strategy B: Use the specific h5py engine
        import h5py
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print("Strategy B Success: Model loaded with safe_mode=False.")
    except Exception as e2:
        print(f"Strategy B failed: {e2}")

@app.get("/")
def home():
    return {
        "status": "IDS API is running", 
        "model_loaded": model is not None,
        "assets_loaded": all([scaler, pca, le])
    }

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model is not loaded. Check Render logs for specific Keras/H5 errors."}
    try:
        df = pd.DataFrame([data])
        scaled_data = scaler.transform(df)
        pca_data = pca.transform(scaled_data)
        prediction = model.predict(pca_data)
        class_idx = np.argmax(prediction)
        return {
            "prediction": str(le.classes_[class_idx]),
            "confidence": float(np.max(prediction))
        }
    except Exception as e:
        return {"error": str(e)}
