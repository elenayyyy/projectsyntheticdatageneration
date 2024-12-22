import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# App Title
st.title('☀️ Solar Power Simulator with Real Data')

st.write("""
This app uses a real-world dataset to analyze and model solar power generation.  
The dataset contains information about solar energy generation and weather parameters, enabling predictions of power output.
""")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload & Overview", "Visualization", "Modeling", "Evaluation"])

# Load dataset
@st.cache
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Data Upload & Overview Tab
with tab1:
    st.header("1️⃣ Data Upload & Overview")
    st.write("Upload the Kaggle Solar Power Generation dataset.")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file)
        st.write("Data Loaded Successfully!")
        
        # Show sample data
        st.write("Sample Data:")
        st.dataframe(data.head())

        # Save data to session state
        st.session_state['data'] = data
        
        # Display basic statistics
        st.write("Data Statistics:")
        st.write(data.describe())
    else:
        st.write("Please upload the dataset to proceed.")

# Visualization Tab
with tab2:
    st.header("2️⃣ Visualization")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Visualize the dataset to uncover relationships and trends.")
        
        # Select features for pairplot
        st.write("Select Features for Pairplot:")
        selected_features = st.multiselect("Choose features:", data.columns, default=data.columns[:3])
        if len(selected_features) > 1:
            pairplot_fig = sns.pairplot(data[selected_features])
            st.pyplot(pairplot_fig)
        
        # Select a feature for distribution plot
        st.write("Feature Distribution:")
        feature = st.selectbox("Select Feature to Plot", data.columns)
        hist_fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, ax=ax)
        st.pyplot(hist_fig)
    else:
        st.write("Please upload the dataset in the 'Data Upload & Overview' tab.")

# Modeling Tab
with tab3:
    st.header("3️⃣ Modeling")
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # Feature selection
        features = st.multiselect("Select Input Features", data.columns, default=data.columns[:-1])
        target = st.selectbox("Select Target Feature", data.columns, index=len(data.columns)-1)

        if features and target:
            X = data[features]
            y = data[target]
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train the model
            st.write("Training a Random Forest model...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Save the model and data
            st.session_state['model'] = rf_model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            st.write("Model trained successfully!")
    else:
        st.write("Please upload the dataset in the 'Data Upload & Overview' tab.")

# Evaluation Tab
with tab4:
    st.header("4️⃣ Evaluation")
    if 'model' in st.session_state:
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        
        # Plot actual vs predicted
        st.write("Actual vs Predicted Power Output:")
        eval_fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax.set_xlabel("Actual Power Output")
        ax.set_ylabel("Predicted Power Output")
        st.pyplot(eval_fig)
    else:
        st.write("Please train a model in the 'Modeling' tab first.")
