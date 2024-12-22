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
num_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=10000, value=1000)
st.sidebar.header("Sample Size & Train/Test Split Configuration")

# Test size slider, as percentage
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30) / 100.0
train_size = 1 - test_size

# Display selected split values
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

    # List of models to train
    models = {
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "SVC": SVC(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "LinearSVC": LinearSVC(random_state=42),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
        "RidgeClassifier": RidgeClassifier(),
        "MultinomialNB": MultinomialNB()
    }

    # Dictionary to store metrics for each model
    model_metrics = {}

    # Train each model and store metrics
    for model_name, model in models.items():
        start_time = time()
        model.fit(X_train, y_train)
        training_time = time() - start_time

        # Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = classification_report(y_test, y_pred, output_dict=True)["accuracy"], classification_report(y_test, y_pred, output_dict=True)["precision"], classification_report(y_test, y_pred, output_dict=True)["recall"], classification_report(y_test, y_pred, output_dict=True)["f1-score"]
        
        # Store metrics
        model_metrics[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "Training Time": training_time
        }

    # Save models to session state to persist across runs
    st.session_state["model_metrics"] = model_metrics
    st.session_state["models"] = models

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
    fig, ax = plt.subplots()
    ax.scatter(data[features[0]], data[features[1]], c=data['Class'].map({"Low": "blue", "Medium": "orange", "High": "green"}), alpha=0.7)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    st.pyplot(fig)

# Performance Metrics Summary
if "model_metrics" in st.session_state:
    st.write("### Performance Metrics Summary")
    selected_models = st.multiselect("Select models to compare", list(st.session_state["model_metrics"].keys()))

    # Model Performance Metrics Comparison Barplot
    if selected_models:
        metrics_data = []
        for model in selected_models:
            metrics_data.append({
                "Model": model,
                "Accuracy": st.session_state["model_metrics"][model]["Accuracy"],
                "Precision": st.session_state["model_metrics"][model]["Precision"],
                "Recall": st.session_state["model_metrics"][model]["Recall"],
                "F1 Score": st.session_state["model_metrics"][model]["F1 Score"]
            })
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index('Model', inplace=True)

        st.write("**Model Performance Metrics Comparison**")
        st.bar_chart(metrics_df)

    # Model Performance Metrics Comparison Barplot
    if selected_models:
        metrics_data = []
        for model in selected_models:
            metrics_data.append({
                "Model": model,
                "Accuracy": np.random.rand(),  # Replace with real accuracy values
                "Precision": np.random.rand(),  # Replace with real precision values
                "Recall": np.random.rand(),  # Replace with real recall values
                "F1 Score": np.random.rand()  # Replace with real F1 score values
            })
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index('Model', inplace=True)

        st.write("**Model Performance Metrics Comparison**")
        st.bar_chart(metrics_df)

    # Saved Models
    st.write("### Saved Models")

    # Saved Models download buttons
    st.write("Download Models:")

    # Create a BytesIO stream for the ExtraTreesClassifier model
    model_io = io.BytesIO()
    joblib.dump(clf, model_io)  # Serialize the model to the BytesIO object
    model_io.seek(0)  # Rewind the BytesIO object to the beginning so it can be read

    # Provide the download button for the ExtraTreesClassifier model
    st.download_button(
    label="Download ExtraTreesClassifier Model",
    data=model_io,
    file_name="ExtraTreesClassifier.pkl",
    mime="application/octet-stream"
    )

    # Confusion Matrices
    st.write("### Confusion Matrices")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
