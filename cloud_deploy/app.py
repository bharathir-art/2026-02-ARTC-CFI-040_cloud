import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

from fastapi import FastAPI
import joblib
import tf_keras as keras
import numpy as np

app = FastAPI()

# Point to the subfolder where the assets actually are
SUB_DIR = "cloud_deploy"

try:
    scaler = joblib.load(os.path.join(SUB_DIR, 'scaler.joblib'))
    pca = joblib.load(os.path.join(SUB_DIR, 'pca_model.joblib'))
    le = joblib.load(os.path.join(SUB_DIR, 'label_encoder.joblib'))
    
    model_path = os.path.join(SUB_DIR, 'ids_model.h5')
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model_status = True
except Exception as e:
    print(f"Loading Error: {e}")
    model = None
    model_status = False

@app.get("/")
def home():
    return {
        "status": "IDS API is running",
        "model_loaded": model_status,
        "debug_info": {
            "current_dir": os.getcwd(),
            "folder_contents": os.listdir(SUB_DIR) if os.path.exists(SUB_DIR) else "Folder not found"
        }
    }
