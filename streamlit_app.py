import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Synthetic Data Generation Function
def generate_synthetic_data(sample_size):
    # Generating synthetic features
    np.random.seed(42)
    soil_type = np.random.choice(['Loamy', 'Sandy', 'Clay'], size=sample_size)
    sunlight_hours = np.random.normal(loc=8, scale=1, size=sample_size)
    water_frequency = np.random.normal(loc=5, scale=2, size=sample_size)
    temperature = np.random.normal(loc=25, scale=2, size=sample_size)
    humidity = np.random.normal(loc=75, scale=5, size=sample_size)
    
    # Generating synthetic growth milestones (as target labels)
    growth_milestone = np.random.choice([0.0, 0.1, 0.8], size=sample_size)  # 0.0 = Seedling, 0.1 = Early Growth, 0.8 = Mature Plant
    
    # Creating DataFrame
    data = pd.DataFrame({
        'Soil_Type': soil_type,
        'Sunlight_Hours': sunlight_hours,
        'Water_Frequency': water_frequency,
        'Temperature': temperature,
        'Humidity': humidity,
        'Growth_Milestone': growth_milestone
    })
    
    return data

# Load and preprocess the data
def preprocess_data(data):
    # Convert categorical feature 'Soil_Type' into numeric values
    data['Soil_Type'] = data['Soil_Type'].map({'Loamy': 1, 'Sandy': 2, 'Clay': 3})
    
    # Features and target variable
    X = data.drop('Growth_Milestone', axis=1)
    y = data['Growth_Milestone']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the model
def train_model(X_train, y_train):
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm

# Visualize Confusion Matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Seedling', 'Early Growth', 'Mature Plant'], yticklabels=['Seedling', 'Early Growth', 'Mature Plant'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot()

# Streamlit App UI
st.title("Water Quality Testing - Plant Growth Milestone Prediction")

# Sidebar Configuration
st.sidebar.header("Data Source")
data_source = st.sidebar.selectbox("Choose Data Source", ["Generate Synthetic Data", "Upload Dataset"])

if data_source == "Generate Synthetic Data":
    sample_size = st.sidebar.slider("Number of samples", 1000, 10000, 5000)
    data = generate_synthetic_data(sample_size)
    st.write(f"Generated Synthetic Data Sample ({sample_size} samples):")
    st.dataframe(data.head())

    # Model Training and Evaluation
    if st.sidebar.button("Train Model"):
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
        model = train_model(X_train, y_train)
        accuracy, report, cm = evaluate_model(model, X_test, y_test)
        
        st.write(f"Model Accuracy: {accuracy:.4f}")
        st.text("Classification Report:")
        st.text(report)
        
        st.write("Confusion Matrix:")
        plot_confusion_matrix(cm)

elif data_source == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Sample:")
        st.dataframe(data.head())

        # Preprocessing and Model Training
        if st.sidebar.button("Train Model"):
            X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
            model = train_model(X_train, y_train)
            accuracy, report, cm = evaluate_model(model, X_test, y_test)
            
            st.write(f"Model Accuracy: {accuracy:.4f}")
            st.text("Classification Report:")
            st.text(report)
            
            st.write("Confusion Matrix:")
            plot_confusion_matrix(cm)

# Learning Curves (Optional, if desired)
st.sidebar.header("Learning Curves")
# If you have learning curves logic, you can include the code here to plot the learning curves.

