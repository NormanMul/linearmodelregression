import streamlit as st
import numpy as np
import joblib

@st.cache(allow_output_mutation=True)
def load_model_scaler():
    data = joblib.load('model_scaler.joblib')
    return data['model'], data['scaler']

model, scaler = load_model_scaler()

st.title('Trust Score Prediction App')
st.markdown('### Made by Naufal Prawiro')

col1, col2 = st.columns([1, 1])
with col1:
    age = st.number_input('Age [A]', min_value=0.0, format="%.4f")
    tcpa_match = st.number_input('% TCPA Match [B]', min_value=0.0, format="%.4f")
    ad_copy_match = st.number_input('% Ad Copy Match [C]', min_value=0.0, format="%.4f")
with col2:
    distance_factor = st.number_input('Distance Factor [D]', min_value=0.0, format="%.4f")
    rate_of_lead_ingestion = st.number_input('Rate of Lead Ingestion [E]', min_value=0.0, format="%.4f")

if st.button('Predict Trust Score'):
    input_data = np.array([[age, tcpa_match, ad_copy_match, distance_factor, rate_of_lead_ingestion]])
    prediction = model.predict(input_data)
    # Ensure that the prediction is transformed using the saved scaler
    trust_score = scaler.transform(prediction.reshape(-1, 1))
    trust_score = np.clip(trust_score, 1, 10)  # Ensure scores remain within 1-10
    st.markdown(f"## Predicted Trust Score: `{round(trust_score[0][0], 2)}`")

st.markdown("### Instructions")
st.markdown("""
- Enter the required fields to predict the Trust Score.
- Click on 'Predict Trust Score' to see the output.
""")
