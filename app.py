import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# --- REBUILD THE DENSE LAYER TO IGNORE THE ERROR ---
class FixedDense(Dense):
    @classmethod
    def from_config(cls, config):
        # This is the line that kills the error
        config.pop('quantization_config', None)
        return super().from_config(config)

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'ids_model.h5')
    
    # Use custom_objects to force Keras to use our 'Fixed' version
    model = load_model(model_path, custom_objects={'Dense': FixedDense})
    
    scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
    pca = joblib.load(os.path.join(base_path, 'pca_model.joblib'))
    le = joblib.load(os.path.join(base_path, 'label_encoder.joblib'))
    return model, scaler, pca, le

model, scaler, pca, le = load_assets()
# ... rest of your code
# ... rest of your code

# The rest of your app logic follows...

model, scaler, pca, le = load_assets()
# ... rest of your code ...
# --- UI Header ---
st.title("🛡️ Network Intrusion Detection System")
st.sidebar.header("Input Settings")
input_mode = st.sidebar.radio("Select Input Method:", ("Bulk CSV Upload", "Manual Packet Entry"))

# Constants from your training script
JITTER_STRENGTH = 0.22
CONFIDENCE_PENALTY = 0.82

# --- Logic for CSV Upload ---
if input_mode == "Bulk CSV Upload":
    st.subheader("📁 Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSE-CIC-IDS2018 formatted CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory=False)
        # Cleaning logic matching your script
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # Drop non-feature columns if they exist
        X = df.drop(columns=['Label', 'Timestamp'], errors='ignore')
        
        if st.button("Run Bulk Prediction"):
            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)
            
            # Fuzzy Inference
            X_fuzzy = X_pca + np.random.normal(0, JITTER_STRENGTH, X_pca.shape)
            y_probs = model.predict(X_fuzzy)
            y_probs[:, 2] *= CONFIDENCE_PENALTY 
            
            preds = np.argmax(y_probs, axis=1)
            df['Prediction'] = le.inverse_transform(preds)
            
            st.write(df[['Prediction']].join(X.head(20)))
            st.bar_chart(df['Prediction'].value_counts())

# --- Logic for Manual Entry ---
else:
    st.subheader("⌨️ Single Packet Manual Entry")
    st.info("Provide core network metrics to check for attack signatures.")
    
    # We create a form to prevent the app from rerunning on every single number change
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            f_dur = st.number_input("Flow Duration", min_value=0.0, value=1000.0)
            tot_f_pkts = st.number_input("Tot Fwd Pkts", min_value=0.0, value=2.0)
            tot_b_pkts = st.number_input("Tot Bwd Pkts", min_value=0.0, value=1.0)
            
        with col2:
            f_iat_tot = st.number_input("Fwd IAT Tot", min_value=0.0, value=500.0)
            f_iat_max = st.number_input("Fwd IAT Max", min_value=0.0, value=100.0)
            pkt_len_max = st.number_input("Pkt Len Max", min_value=0.0, value=50.0)
            
        with col3:
            flow_byts_s = st.number_input("Flow Byts/s", min_value=0.0, value=0.0)
            flow_pkts_s = st.number_input("Flow Pkts/s", min_value=0.0, value=0.0)
            init_fwd_win = st.number_input("Init Fwd Win Byts", min_value=0.0, value=255.0)

        submit = st.form_submit_button("Analyze Single Packet")

    if submit:
        # 1. Create a zero-filled array matching the training feature count
        input_array = np.zeros((1, scaler.n_features_in_))
        
        # 2. Map the manual inputs to the array (assuming standard column order)
        # Note: You should ideally use column names if you saved them, 
        # but here we fill the first 9 positions as an example:
        input_array[0, 0:9] = [f_dur, tot_f_pkts, tot_b_pkts, f_iat_tot, f_iat_max, pkt_len_max, flow_byts_s, flow_pkts_s, init_fwd_win]
        
        # 3. Processing Pipeline
        scaled_data = scaler.transform(input_array)
        pca_data = pca.transform(scaled_data)
        
        # 4. Model Prediction
        y_probs = model.predict(pca_data)
        y_probs[:, 2] *= CONFIDENCE_PENALTY 
        result_idx = np.argmax(y_probs, axis=1)
        result_label = le.inverse_transform(result_idx)[0]
        
        # 5. Display Result
        if result_label == "Benign":
            st.balloons()
            st.success(f"✅ Prediction: {result_label} (Traffic appears safe)")
        else:
            st.error(f"🚨 ALERT: {result_label} Detected!")
            st.warning("Confidence score adjusted by Fuzzy Inference Engine.")