import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

from fastapi import FastAPI
import joblib
import tf_keras as keras

app = FastAPI()

# Strategy: Define paths relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(name):
    return os.path.join(CURRENT_DIR, name)

# --- LOADING ---
try:
    scaler = joblib.load(load_file('scaler.joblib'))
    pca = joblib.load(load_file('pca_model.joblib'))
    le = joblib.load(load_file('label_encoder.joblib'))
    
    # Load Model using the tf_keras bridge
    model = keras.models.load_model(load_file('ids_model.h5'), compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model_status = True
except Exception as e:
    print(f"DEPLOYMENT ERROR: {e}")
    model = None
    model_status = False

@app.get("/")
def home():
    return {
        "status": "IDS API is running", 
        "model_loaded": model_status,
        "files_in_dir": os.listdir(CURRENT_DIR) # This helps us debug if files are missing
    }
