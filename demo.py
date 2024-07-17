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
    # Drop rows with missing target values
    data = data.dropna(subset=['Trust Score (Normalized)'])

    # Handle non-numeric columns by encoding them
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    
    # Select only the specified features
    features = ['Age [A]', '% TCPA Match [B]', '% Ad Copy Match [C]', 'Distance Factor [D]', 'Rate of Lead Ingestion [E]']
    X = data[features]
    
    # Scale the target to be between 1 and 10
    y = data['Trust Score (Normalized)']
    scaler = MinMaxScaler(feature_range=(1, 10))
    y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    print("Data preprocessed successfully.")
    return X, y

def train_model(X, y):
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

def save_model(model, file_path):
    print("Saving model...")
    joblib.dump(model, file_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    try:
        data = load_data('dataframe.csv')  # Ensure this file is in the root directory of the project
        X, y = preprocess_data(data)
        print("Features used for training:", X.columns.tolist())  # Print the features used for training
        model = train_model(X, y)
        save_model(model, 'model.joblib')
        print("All steps completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
