import os
# Force Keras 3 behavior to handle 'quantization_config'
os.environ["TF_USE_LEGACY_KERAS"] = "0"

from fastapi import FastAPI
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

app = FastAPI()

# --- 1. INITIALIZE VARIABLES ---
model = None
scaler = None
pca = None
le = None

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 2. ASSET & MODEL LOADING ---
try:
    # Load scikit-learn assets
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
    pca = joblib.load(os.path.join(BASE_DIR, 'pca_model.joblib'))
    le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.joblib'))
    print("Pre-processing assets loaded successfully!")

    # Load TensorFlow model
    model_path = os.path.join(BASE_DIR, 'ids_model.h5')
    # compile=False is used to skip training-specific config errors
    model = tf.keras.models.load_model(model_path, compile=False)
    # Re-compile for inference
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print("AI Model loaded successfully!")

except Exception as e:
    print(f"Startup Error: {e}")

@app.get("/")
def home():
    return {
        "status": "IDS API is running", 
        "model_loaded": model is not None,
        "assets_loaded": all([scaler, pca, le]),
        "version": "2.0-keras3-fix"
    }

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded. Check server logs for Keras/TensorFlow version issues."}
    
    try:
        # 1. Convert incoming JSON to DataFrame
        df = pd.DataFrame([data])
        
        # 2. Scale features
        scaled_data = scaler.transform(df)
        
        # 3. Apply Dimensionality Reduction (PCA)
        pca_data = pca.transform(scaled_data)
        
        # 4. Generate Prediction
        prediction = model.predict(pca_data)
        class_idx = np.argmax(prediction)
        
        # 5. Map index back to original label (e.g., 'Normal', 'DoS')
        result_label = str(le.classes_[class_idx])
        confidence = float(np.max(prediction))
        
        return {
            "prediction": result_label,
            "confidence": f"{confidence:.2%}",
            "status": "success"
        }
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}
