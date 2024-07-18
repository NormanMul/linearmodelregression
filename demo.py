import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

def load_data(file_path):
    print("Loading data...")
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return data

def preprocess_data(data):
    print("Preprocessing data...")
    data = data.dropna(subset=['Trust Score (Normalized)'])
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    features = ['Age [A]', '% TCPA Match [B]', '% Ad Copy Match [C]', 'Distance Factor [D]', 'Rate of Lead Ingestion [E]']
    X = data[features]
    y = data['Trust Score (Normalized)']
    return X, y

def train_model(X, y):
    scaler = MinMaxScaler(feature_range=(1, 10))
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    model = LinearRegression()
    model.fit(X, y_scaled)
    return model, scaler

def save_model_scaler(model, scaler):
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    data = load_data('dataframe.csv')
    X, y = preprocess_data(data)
    model, scaler = train_model(X, y)
    save_model_scaler(model, scaler)
