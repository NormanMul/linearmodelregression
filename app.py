import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Function to load the model and scaler
@st.cache(allow_output_mutation=True)
def load_resources():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# Load model and scaler
model, scaler = load_resources()

# Streamlit app layout
st.title('Trust Score Prediction App')
st.markdown('### Made by Naufal Prawiro')

# Centralizing the prediction inputs and button
col1, col2 = st.columns([1, 1])
with col1:
    age = st.number_input('Age [A]', min_value=0.0, format="%.4f")
    tcpa_match = st.number_input('% TCPA Match [B]', min_value=0.0, format="%.4f")
    ad_copy_match = st.number_input('% Ad Copy Match [C]', min_value=0.0, format="%.4f")
with col2:
    distance_factor = st.number_input('Distance Factor [D]', min_value=0.0, format="%.4f")
    rate_of_lead_ingestion = st.number_input('Rate of Lead Ingestion [E]', min_value=0.0, format="%.4f")

# Button to predict score
if st.button('Predict Trust Score', help='Click to predict the Trust Score based on input parameters'):
    input_data = np.array([[age, tcpa_match, ad_copy_match, distance_factor, rate_of_lead_ingestion]])
    prediction = model.predict(input_data)
    trust_score = round(scaler.transform(prediction.reshape(-1, 1))[0][0], 2)
    st.markdown(f"## Predicted Trust Score: `{trust_score}`")

# Additional information or instructions can be added below
st.markdown("### Instructions")
st.markdown("""
- Enter the required fields to predict the Trust Score.
- Click on 'Predict Trust Score' to see the output.
""")
