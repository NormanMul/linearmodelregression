import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('model.joblib')
    return model

model = load_model()

# Streamlit app
st.title('Trust Score Prediction App')

# Sidebar
st.sidebar.title("Made by Naufal Prawironegoro")

st.header("Input the Parameters")

# Input fields for each parameter
age = st.number_input('Age [A]', min_value=0.0, format="%.4f")
tcpa_match = st.number_input('% TCPA Match [B]', min_value=0.0, format="%.4f")
ad_copy_match = st.number_input('% Ad Copy Match [C]', min_value=0.0, format="%.4f")
distance_factor = st.number_input('Distance Factor [D]', min_value=0.0, format="%.4f")
rate_of_lead_ingestion = st.number_input('Rate of Lead Ingestion [E]', min_value=0.0, format="%.4f")

# Prediction button
if st.button('Predict Trust Score'):
    input_data = np.array([[age, tcpa_match, ad_copy_match, distance_factor, rate_of_lead_ingestion]])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
        trust_score = round(prediction[0], 2)
        st.write('Predicted Trust Score:', trust_score)
    except Exception as e:
        st.write('Error:', str(e))
