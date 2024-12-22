import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# App Title
st.title("Water Quality Testing - Plant Growth Milestone Prediction")

# Sidebar for Data Upload
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose Data Source:", ["Upload Dataset"])

if data_source == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset:")
        st.dataframe(data.head())

        # Train-Test Split Configuration
        st.sidebar.header("Train-Test Split Configuration")
        test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30) / 100.0

        # Ensure data has necessary columns (Soil_Type, Sunlight_Hours, Water_Frequency, Temperature, Humidity, Growth_Milestone)
        if all(col in data.columns for col in ['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Temperature', 'Humidity', 'Growth_Milestone']):
            # Preprocessing
            features = ['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Temperature', 'Humidity']
            X = data[features]
            y = data['Growth_Milestone']

            # Convert categorical feature 'Soil_Type' into numeric values if necessary
            X['Soil_Type'] = X['Soil_Type'].map({'Loamy': 1, 'Sandy': 2, 'Clay': 3})

            # Standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

            # Train the Random Forest Model
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Model Evaluation
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            st.write("### Model Performance")
            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write("**Classification Report:**")
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Seedling', 'Early Growth', 'Mature Plant'], yticklabels=['Seedling', 'Early Growth', 'Mature Plant'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

            # Feature Visualization
            st.write("### Feature Visualization")
            plot_type = st.selectbox("Select Plot Type:", ["2D Plot", "3D Plot", "Correlation Matrix", "Line Plot"])

            if plot_type == "2D Plot":
                x_feature = st.selectbox("Select X-axis feature:", features)
                y_feature = st.selectbox("Select Y-axis feature:", features)
                fig, ax = plt.subplots()
                ax.scatter(data[x_feature], data[y_feature], c=data['Growth_Milestone'], cmap='viridis', alpha=0.7)
                ax.set_xlabel(x_feature)
                ax.set_ylabel(y_feature)
                st.pyplot(fig)

            elif plot_type == "3D Plot":
                if len(features) < 3:
                    st.warning("Need at least 3 features for 3D Plot.")
                else:
                    x_feature = st.selectbox("Select X-axis feature:", features, key="3d_x")
                    y_feature = st.selectbox("Select Y-axis feature:", features, key="3d_y")
                    z_feature = st.selectbox("Select Z-axis feature:", features, key="3d_z")
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(data[x_feature], data[y_feature], data[z_feature], c=data['Growth_Milestone'], cmap='viridis', alpha=0.7)
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(y_feature)
                    ax.set_zlabel(z_feature)
                    st.pyplot(fig)

            elif plot_type == "Correlation Matrix":
                fig, ax = plt.subplots()
                sns.heatmap(data[features].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

            elif plot_type == "Line Plot":
                fig, ax = plt.subplots()
                for feature in features:
                    ax.plot(data[feature], label=feature)
                ax.legend()
                st.pyplot(fig)

            # Download Options
            st.write("### Download Options")
            data_csv = data.to_csv(index=False)
            st.download_button("Download Dataset", data_csv, "dataset.csv", "text/csv")

            # Save and download the trained model
            model_file = "trained_model.pkl"
            joblib.dump(clf, model_file)
            st.download_button("Download Trained Model", model_file, file_name="trained_model.pkl")
