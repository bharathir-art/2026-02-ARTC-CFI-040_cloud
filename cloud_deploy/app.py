import os
# Ensure we are NOT using legacy keras so Keras 3 can load the model
os.environ["TF_USE_LEGACY_KERAS"] = "0" 

from fastapi import FastAPI
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd

app = FastAPI()

# Your files are in /app, so we use current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
scaler = None
pca = None
le = None

try:
    # Load scikit-learn assets
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
    pca = joblib.load(os.path.join(BASE_DIR, 'pca_model.joblib'))
    le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.joblib'))
    
    # Load Model with Keras 3 engine
    model_path = os.path.join(BASE_DIR, 'ids_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print("SUCCESS: Model and assets loaded!")
except Exception as e:
    print(f"Loading Error: {e}")

@app.get("/")
def home():
    return {
        "status": "IDS API is running",
        "model_loaded": model is not None,
        "assets_loaded": all([scaler, pca, le]),
        "files_verified": os.listdir(BASE_DIR)
    }

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}
    try:
        df = pd.DataFrame([data])
        scaled = scaler.transform(df)
        pca_data = pca.transform(scaled)
        pred = model.predict(pca_data)
        class_idx = np.argmax(pred)
        return {
            "prediction": str(le.classes_[class_idx]),
            "confidence": float(np.max(pred))
        }
    except Exception as e:
        return {"error": str(e)}
