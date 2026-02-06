import os
import joblib
import numpy as np

# Force the Keras legacy bridge
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as keras
from fastapi import FastAPI

app = FastAPI()

# Automatically find the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
model_loaded = False

try:
    # Load assets using absolute paths
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
    pca = joblib.load(os.path.join(BASE_DIR, 'pca_model.joblib'))
    le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.joblib'))
    
    # Load Model
    model = keras.models.load_model(os.path.join(BASE_DIR, 'ids_model.h5'), compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model_loaded = True
    print("SUCCESS: All files loaded from " + BASE_DIR)
except Exception as e:
    print(f"CRITICAL ERROR during loading: {e}")

@app.get("/")
def home():
    return {
        "status": "IDS API is running",
        "model_loaded": model_loaded,
        "base_directory": BASE_DIR,
        "files_visible": os.listdir(BASE_DIR)
    }
