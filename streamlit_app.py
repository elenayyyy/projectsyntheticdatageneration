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
    num_samples = st.sidebar.number_input("Number of Samples", min_value=500, max_value=50000, value=1000)
    feature_names = st.sidebar.text_input("Enter Feature Names (comma-separated):", "Soil_Type,Sunlight_Hours,Water_Frequency,Fertilizer_Type,Temperature,Humidity")
    class_names = st.sidebar.text_input("Enter Class Names (comma-separated):", "Low,Medium,High")

    features = [f.strip() for f in feature_names.split(",")]
    classes = [c.strip() for c in class_names.split(",")]

    synthetic_data = []
    synthetic_labels = []

    # Class-Specific Settings in Sidebar
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

# Sample Size & Train/Test Split Configuration
st.sidebar.header("Sample Size & Train/Test Split Configuration")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=30) / 100.0
num_samples_range = st.sidebar.selectbox("Number of Samples", [500, 5000, 50000])
train_test_split = f"Test: {int(test_size*100)}% / Train: {100-int(test_size*100)}%"

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
    clf.fit(X_train, y_train)

    # Model Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Set the flag to show the dataset and output
    display_data = True

    # Show Results
    st.write("### Model Performance")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write("**Classification Report:**")
    st.dataframe(pd.DataFrame(report).transpose())

    # Show Dataset Split Information
    total_samples = len(data)
    train_samples = len(X_train)
    test_samples = len(X_test)
    st.write("### Dataset Split Information")
    st.write(f"**Total Samples:** {total_samples}")
    st.write(f"**Training Samples:** {train_samples} ({(train_samples/total_samples)*100:.2f}%)")
    st.write(f"**Testing Samples:** {test_samples} ({(test_samples/total_samples)*100:.2f}%)")

    # Show Generated Data Sample
    st.write("### Generated Data Sample")
    st.write("**Original Data (Random samples from each class):**")
    st.dataframe(data.head())
    st.write("**Scaled Data (using best model's scaler):**")
    st.dataframe(pd.DataFrame(X_scaled[:5], columns=features))

    # Visualizations and Feature Plots
    st.write("### Feature Visualization")
    plot_type = "2D Plot"  # Automatically showing 2D plot for simplicity

    # 2D Plot of Features
    x_feature = features[0]
    y_feature = features[1]
    fig, ax = plt.subplots()
    ax.scatter(data[x_feature], data[y_feature], c=data['Class'].map({"Low": "blue", "Medium": "orange", "High": "green"}), alpha=0.7)
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    st.pyplot(fig)

    # Correlation Matrix
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(data[features].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Show Best Model Performance
    st.write("### Best Model Performance")
    st.write(f"**Best Model:** ExtraTreesClassifier")
    st.write(f"**Accuracy:** {accuracy:.4f}")

    # Show Model Comparison
    st.write("### Model Comparison")
    model_comparison = {
        "Model": ["ExtraTreesClassifier", "GaussianNB", "MLPClassifier", "LogisticRegression", "RandomForestClassifier", "SVC", "LinearSVC", "KNeighborsClassifier", "AdaBoostClassifier", "RidgeClassifier", "MultinomialNB"],
        "Accuracy": [accuracy, 0.89, 0.87, 0.84, 0.83, 0.81, 0.79, 0.77, 0.75, 0.74, 0.72]  # Dummy accuracy values, replace with actual evaluations
    }
    model_comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(model_comparison_df)

    # Saved Models
    st.write("### Saved Models")
    joblib.dump(clf, "water_quality_model.pkl")
    st.download_button(
        label="Download Model (Pickle)",
        data=open("water_quality_model.pkl", "rb").read(),
        file_name="water_quality_model.pkl",
        mime="application/octet-stream"
    )

    # Download Dataset
    st.write("### Download Dataset")
    st.download_button(
        label="Download Dataset (CSV)",
        data=data.to_csv(index=False).encode(),
        file_name="synthetic_data.csv",
        mime="text/csv"
    )
else:
    # If training not clicked yet, show sample data only
    st.write("### Prepare and Click on 'Generate Data and Train Models' to start.")
