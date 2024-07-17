import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    data = joblib.load('model.joblib')
    return data['model'], data['scaler']

model, scaler = load_model()

st.title('Trust Score Prediction App')
st.sidebar.title("Input Parameters")

age = st.sidebar.number_input('Age [A]', min_value=0.0, format="%.4f")
tcpa_match = st.sidebar.number_input('% TCPA Match [B]', min_value=0.0, format="%.4f")
ad_copy_match = st.sidebar.number_input('% Ad Copy Match [C]', min_value=0.0, format="%.4f")
distance_factor = st.sidebar.number_input('Distance Factor [D]', min_value=0.0, format="%.4f")
rate_of_lead_ingestion = st.sidebar.number_input('Rate of Lead Ingestion [E]', min_value=0.0, format="%.4f")

if st.sidebar.button('Predict Trust Score'):
    input_data = np.array([[age, tcpa_match, ad_copy_match, distance_factor, rate_of_lead_ingestion]])
    prediction = model.predict(input_data)
    trust_score = round(scaler.transform(prediction.reshape(-1, 1))[0][0], 2)
    st.write('Predicted Trust Score:', trust_score)
