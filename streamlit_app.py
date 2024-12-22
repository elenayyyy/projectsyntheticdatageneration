import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
from time import time
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

    num_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=10000, value=1000)

    st.sidebar.subheader("Class-Specific Settings")
    class_settings = {}
    for cls in classes:
        class_settings[cls] = {}
        st.sidebar.subheader(f"{cls} Settings")
        for feature in features:
            mean = st.sidebar.number_input(f"Mean for {feature} ({cls})", value=50.0, key=f"{cls}_{feature}_mean")
            std = st.sidebar.number_input(f"Std Dev for {feature} ({cls})", value=10.0, key=f"{cls}_{feature}_std")
            class_settings[cls][feature] = (mean, std)

        for _ in range(num_samples // len(classes)):
            synthetic_data.append([np.random.normal(class_settings[cls][f][0], class_settings[cls][f][1]) for f in features])
            synthetic_labels.append(cls)

    data = pd.DataFrame(synthetic_data, columns=features)
    data['Class'] = synthetic_labels

st.sidebar.header("Sample Size & Train/Test Split Configuration")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30) / 100.0
train_size = 1 - test_size

st.sidebar.write(f"Test: {test_size * 100}% / Train: {train_size * 100}%")

start_training = st.sidebar.button("Generate Data and Train Models")

if start_training:
    if data.empty:
        st.warning("Please upload or generate data before training models.")
    else:
        X = data[features]
        y = data['Class']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

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

        model_metrics = {}

        for model_name, model in models.items():
            start_time = time()
            model.fit(X_train, y_train)
            training_time = time() - start_time

            if "trained_models" not in st.session_state:
                st.session_state["trained_models"] = {}
            st.session_state["trained_models"][model_name] = model

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report["macro avg"]["precision"]
            recall = report["macro avg"]["recall"]
            f1_score = report["macro avg"]["f1-score"]
            accuracy = accuracy_score(y_test, y_pred)

            model_metrics[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score,
                "Training Time": training_time
            }

        st.session_state["model_metrics"] = model_metrics

        st.write("### Dataset Split Information")
        total_samples = len(data)
        train_samples = len(X_train)
        test_samples = len(X_test)
        st.write(f"**Total Samples:** {total_samples}")
        st.write(f"**Training Samples:** {train_samples} ({(train_samples/total_samples)*100:.2f}%)")
        st.write(f"**Testing Samples:** {test_samples} ({(test_samples/total_samples)*100:.2f}%)")

        st.write("### Generated Data Sample")
        st.dataframe(data.head())
        st.dataframe(pd.DataFrame(X_scaled[:5], columns=features))

if "model_metrics" in st.session_state:
    st.write("### Performance Metrics Summary")
    selected_models = st.multiselect("Select models to compare", list(st.session_state["model_metrics"].keys()))

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
        st.bar_chart(metrics_df)

if "trained_models" in st.session_state:
    st.write("### Save and Download Trained Models")
    model_name = st.selectbox("Choose a model to save:", list(st.session_state["trained_models"].keys()))

    if st.button("Save Selected Model"):
        selected_model = st.session_state["trained_models"][model_name]
        model_io = io.BytesIO()
        joblib.dump(selected_model, model_io)
        model_io.seek(0)

        st.download_button(
            label=f"Download {model_name} Model",
            data=model_io,
            file_name=f"{model_name}.pkl",
            mime="application/octet-stream"
        )

st.write("### Confusion Matrices")
if "trained_models" in st.session_state:
    for model_name, model in st.session_state["trained_models"].items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
        ax.set_title(f"Confusion Matrix for {model_name}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

# "Saved Models" Table
if "trained_models" in st.session_state:
    st.write("### Saved Models")
    saved_models_data = []
    for model_name, model in st.session_state["trained_models"].items():
        accuracy = st.session_state["model_metrics"].get(model_name, {}).get("Accuracy", "N/A")
        saved_models_data.append({"Model": model_name, "Accuracy": accuracy})

    saved_models_df = pd.DataFrame(saved_models_data)
    st.dataframe(saved_models_df)

# "Download Models" CSV
if "trained_models" in st.session_state:
    st.write("### Download Models CSV")
    download_data = []
    for model_name, model in st.session_state["trained_models"].items():
        accuracy = st.session_state["model_metrics"].get(model_name, {}).get("Accuracy", "N/A")
        download_data.append({"Model": model_name, "Accuracy": accuracy})

    download_df = pd.DataFrame(download_data)
    csv = download_df.to_csv(index=False)
    st.download_button(
        label="Download Models CSV",
        data=csv,
        file_name="models.csv",
        mime="text/csv"
    )
