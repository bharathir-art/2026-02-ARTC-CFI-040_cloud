import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

from fastapi import FastAPI
import joblib
import tf_keras as keras # This matches your requirements.txt
import numpy as np

app = FastAPI()

# Simple loading because we are in the same folder
try:
    scaler = joblib.load('scaler.joblib')
    pca = joblib.load('pca_model.joblib')
    le = joblib.load('label_encoder.joblib')
    
    model = keras.models.load_model('ids_model.h5', compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model_loaded = True
except Exception as e:
    print(f"Error: {e}")
    model = None
    model_loaded = False

@app.get("/")
def home():
    return {
        "status": "IDS API is running",
        "model_loaded": model_loaded,
        "current_directory": os.getcwd(),
        "files_found": os.listdir('.')
    }
