import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# App title
st.title("Food Ingredients and Allergen Modeling and Simulation")

# Sidebar options
st.sidebar.title("Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Generate Synthetic Data", "Upload Dataset"])

# Load dataset or generate synthetic data
if data_source == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV):", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset:")
        st.write(data.head())
    else:
        st.warning("Please upload a dataset to proceed.")
else:
    st.sidebar.title("Synthetic Data Generation")
    feature_names = st.sidebar.text_input("Enter feature names (comma-separated):", "length, width, density, pH")
    class_names = st.sidebar.text_input("Enter class names (comma-separated):", "class1, class2")

    # Class-specific settings
    feature_names = [f.strip() for f in feature_names.split(",")]
    class_names = [c.strip() for c in class_names.split(",")]

    synthetic_data = []
    for cls in class_names:
        st.sidebar.write(f"### {cls} Settings")
        cls_data = {}
        for feature in feature_names:
            mean = st.sidebar.number_input(f"Mean for {feature} ({cls}):", value=100.0)
            std = st.sidebar.number_input(f"Std Dev for {feature} ({cls}):", value=10.0)
            cls_data[feature] = np.random.normal(mean, std, 1000)
        cls_data["class"] = [cls] * 1000
        synthetic_data.append(pd.DataFrame(cls_data))

    data = pd.concat(synthetic_data).reset_index(drop=True)
    st.write("### Generated Synthetic Dataset:")
    st.write(data.head())

# Train/test split
st.sidebar.title("Sample Size & Train/Test Split Configuration")
sample_size = st.sidebar.number_input("Number of samples:", value=len(data))
test_size = st.sidebar.slider("Test size (%):", min_value=10, max_value=50, value=30) / 100

X = data.drop(columns="class")
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
st.write("### Dataset Split Information")
st.write(f"Total Samples: {len(data)}")
st.write(f"Training Samples: {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
st.write(f"Testing Samples: {len(X_test)} ({test_size * 100:.0f}%)")

# Train model
st.sidebar.title("Generate Data and Train Models")
if st.sidebar.button("Train Model"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)

    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names, ax=ax)
    st.pyplot(fig)

# Visualization options
st.sidebar.title("Feature Visualization")
plot_type = st.sidebar.radio("Select plot type:", ["2D Plot", "3D Plot"])
x_axis = st.sidebar.selectbox("Select X-axis feature:", feature_names)
y_axis = st.sidebar.selectbox("Select Y-axis feature:", feature_names)
if plot_type == "3D Plot":
    z_axis = st.sidebar.selectbox("Select Z-axis feature:", feature_names)

if st.sidebar.button("Generate Plot"):
    if plot_type == "2D Plot":
        fig, ax = plt.subplots()
        for cls in class_names:
            subset = data[data["class"] == cls]
            ax.scatter(subset[x_axis], subset[y_axis], label=cls)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.legend()
        st.pyplot(fig)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for cls in class_names:
            subset = data[data["class"] == cls]
            ax.scatter(subset[x_axis], subset[y_axis], subset[z_axis], label=cls)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)
        ax.legend()
        st.pyplot(fig)

# Download options
st.sidebar.title("Download Options")
if st.sidebar.button("Download Dataset"):
    data.to_csv("processed_data.csv", index=False)
    st.sidebar.download_button(
        label="Download Processed Dataset", file_name="processed_data.csv", data=data.to_csv(index=False)
    )

if st.sidebar.button("Download Model"):
    with open("model.pkl", "rb") as f:
        st.sidebar.download_button(label="Download Trained Model", file_name="model.pkl", data=f)
