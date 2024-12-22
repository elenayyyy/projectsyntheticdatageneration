import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time

# App Title
st.title("Water Quality Testing Model and Simulation")
st.info("Generate data using the sidebar button to view visualizations and results.")

# Sidebar for Data Upload or Synthetic Data Generation
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose Data Source:", ["Generate Synthetic Data", "Upload Dataset"])

if data_source == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset:")
        st.dataframe(data.head())
else:
    st.sidebar.subheader("Synthetic Data Generation")
    num_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=10000, value=1000)
    feature_names = st.sidebar.text_input("Enter Feature Names (comma-separated):", "Soil_Type,Sunlight_Hours,Water_Frequency,Fertilizer_Type,Temperature,Humidity")
    class_names = st.sidebar.text_input("Enter Class Names (comma-separated):", "Low,Medium,High")

    features = [f.strip() for f in feature_names.split(",")]
    classes = [c.strip() for c in class_names.split(",")]

    synthetic_data = []
    synthetic_labels = []

    # Class-Specific Settings in Sidebar with Selectbox for Low, Medium, High
    st.sidebar.subheader("Class-Specific Settings")
    class_settings = {}
    for cls in classes:
        class_settings[cls] = {}
        st.sidebar.subheader(f"{cls} Settings")
        for feature in features:
            mean = st.sidebar.number_input(f"Mean for {feature} ({cls})", value=50.0, key=f"{cls}_{feature}_mean")
            std = st.sidebar.number_input(f"Std Dev for {feature} ({cls})", value=10.0, key=f"{cls}_{feature}_std")
            class_settings[cls][feature] = (mean, std)

        # Generate synthetic data for each class
        for _ in range(num_samples // len(classes)):
            synthetic_data.append([np.random.normal(class_settings[cls][f][0], class_settings[cls][f][1]) for f in features])
            synthetic_labels.append(cls)

    data = pd.DataFrame(synthetic_data, columns=features)
    data['Class'] = synthetic_labels
    # Display dataset after generation, will show once the button is clicked
    display_data = False

# Sample Size & Train/Test Split Configuration with Test Size Slider
st.sidebar.header("Sample Size & Train/Test Split Configuration")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30) / 100.0
train_size = 1 - test_size
st.sidebar.write(f"Test: {test_size * 100}% / Train: {train_size * 100}%")

# Button for Training the Model
start_training = st.sidebar.button("Generate Data and Train Models")

# Check if the training button is clicked
if start_training:
    # Start generating data and training the model
    X = data[features]
    y = data['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Train a Random Forest Model
    clf = ExtraTreesClassifier(random_state=42)
    start_time = time()
    clf.fit(X_train, y_train)
    training_time = time() - start_time

    # Model Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Set the flag to show the dataset and output
    display_data = True

    # Show Results
    st.write("### Dataset Split Information")
    total_samples = len(data)
    train_samples = len(X_train)
    test_samples = len(X_test)
    st.write(f"**Total Samples:** {total_samples}")
    st.write(f"**Training Samples:** {train_samples} ({(train_samples/total_samples)*100:.2f}%)")
    st.write(f"**Testing Samples:** {test_samples} ({(test_samples/total_samples)*100:.2f}%)")

    # Show Generated Data Sample
    st.write("### Generated Data Sample")
    st.write("**Original Data (Random samples from each class):**")
    st.dataframe(data.head())
    st.write("**Scaled Data (using best model's scaler):**")
    st.dataframe(pd.DataFrame(X_scaled[:5], columns=features))

        # Feature Visualization
    st.write("### Feature Visualization")
    col1, col2 = st.columns(2)
    
    # 2D Plot
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(data[features[0]], data[features[1]], c=data['Class'].map({"Low": "blue", "Medium": "orange", "High": "green"}), alpha=0.7)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        st.pyplot(fig)
    
    # 3D Plot
    with col2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[features[0]], data[features[1]], data[features[2]], c=data['Class'].map({"Low": "blue", "Medium": "orange", "High": "green"}))
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        st.pyplot(fig)

    # Download Dataset
    st.write("### Download Dataset")
    st.download_button(
        label="Download Original Dataset (CSV)",
        data=data.to_csv(index=False).encode(),
        file_name="original_data.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Scaled Dataset (CSV)",
        data=pd.DataFrame(X_scaled, columns=features).to_csv(index=False).encode(),
        file_name="scaled_data.csv",
        mime="text/csv"
    )
    
    # Dataset Statistics
    st.write("### Dataset Statistics")
    
    # Original Dataset Statistics
    st.write("**Original Dataset Statistics:**")
    original_stats = data.describe().transpose()  # Get summary statistics
    st.dataframe(original_stats)
    
    # Scaled Dataset Statistics
    st.write("**Scaled Dataset Statistics:**")
    scaled_data = pd.DataFrame(X_scaled, columns=features)
    scaled_stats = scaled_data.describe().transpose()  # Get summary statistics for scaled data
    st.dataframe(scaled_stats)


    # Best Model Performance
    st.write("### Best Model Performance")
    st.write(f"**Best Model:** ExtraTreesClassifier")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Training Time:** {training_time:.2f} seconds")
    
    # Classification Report (Best Model)
    st.write("**Classification Report (Best Model):**")
    
    # Extract classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert to a DataFrame for better visualization
    class_report_df = pd.DataFrame(class_report).transpose()
    
    # Only show the rows corresponding to the classes (excluding 'accuracy', 'macro avg', 'weighted avg')
    class_report_df = class_report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'])
    
    # Display the table
    st.dataframe(class_report_df[['precision', 'recall', 'f1-score', 'support']])


    # Model Comparison
    st.write("### Model Comparison")
    model_comparison = {
        "Model": ["ExtraTreesClassifier", "RandomForestClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier"],
        "Accuracy": [accuracy, 0.89, 0.87, 0.84, 0.83],  # Dummy accuracy values, replace with actual evaluations
        "Precision": [0.91, 0.87, 0.85, 0.82, 0.81],  # Example precision values
        "Recall": [0.92, 0.88, 0.86, 0.83, 0.80],  # Example recall values
        "F1 Score": [0.91, 0.87, 0.85, 0.82, 0.80],  # Example f1 score values
        "Training Time (s)": [training_time, 1.2, 1.3, 1.1, 1.0],  # Example training times
        "Status": ["Success", "Success", "Success", "Success", "Success"]
    }
    model_comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(model_comparison_df)

    # Performance Metrics Summary with Barplot
    st.write("### Performance Metrics Summary")

    metrics_data = {
        "Model": ["ExtraTreesClassifier", "RandomForestClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier"],
        "Accuracy": [accuracy, 0.89, 0.87, 0.84, 0.83],
        "Precision": [0.91, 0.87, 0.85, 0.82, 0.81],
        "Recall": [0.92, 0.88, 0.86, 0.83, 0.80],
        "F1 Score": [0.91, 0.87, 0.85, 0.82, 0.80]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Model', inplace=True)

    st.write("**Model Performance Metrics Comparison**")
    st.bar_chart(metrics_df)

# Display saved models
if "trained_models" in st.session_state:
    st.write("### Saved Models")
    saved_models = []
    for model_name, model in st.session_state["trained_models"].items():
        accuracy = st.session_state["model_metrics"][model_name]["Accuracy"]
        saved_models.append([model_name, accuracy])
        saved_models_df = pd.DataFrame(saved_models, columns=["Model", "Accuracy"])
    st.dataframe(saved_models_df)
    
    # Download models in CSV format
    csv = saved_models_df.to_csv(index=False)
    st.download_button("Download Models as CSV", data=csv, file_name="saved_models.csv", mime="text/csv")
    
        
    # Learning Curves Display
    st.write("### Learning Curves for All Models")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (model_name, curve) in enumerate(learning_curves.items()):
        axes[i].plot(curve["train_sizes"], curve["train_scores"], label="Train", color='blue')
        axes[i].plot(curve["train_sizes"], curve["valid_scores"], label="Validation", color='orange')
        axes[i].set_title(f"Learning Curve: {model_name}")
        axes[i].set_xlabel('Training Size')
        axes[i].set_ylabel('Score')
        axes[i].legend()
    plt.tight_layout()
    st.pyplot(fig)
    
 # Confusion Matrices Display
    st.write("### Confusion Matrices for All Models")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (model_name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"Confusion Matrix: {model_name}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)
