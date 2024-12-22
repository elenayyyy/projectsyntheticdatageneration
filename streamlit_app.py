import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generating synthetic data based on given settings
def generate_synthetic_data(samples=10000):
    # Define feature means and std deviations for each class/feature
    data = []
    for _ in range(samples):
        # Soil_Type
        soil_type = np.random.choice(['Loamy', 'Sandy', 'Clay'])
        
        # Sunlight Hours
        sunlight_hours = np.random.normal(8, 1)
        
        # Water Frequency (days)
        water_frequency = np.random.normal(5, 2)
        
        # Temperature (°C)
        temperature = np.random.normal(25, 2)
        
        # Humidity (%)
        humidity = np.random.normal(75, 5)
        
        # Growth Milestone (target label: Seedling, Early Growth, Mature Plant)
        growth_milestone = np.random.choice([0, 0.4, 0.8])  # 0: Seedling, 0.4: Early Growth, 0.8: Mature
        
        # Collect the row of data
        row = [soil_type, sunlight_hours, water_frequency, temperature, humidity, growth_milestone]
        data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Temperature', 'Humidity', 'Growth_Milestone'])
    return df

# Load or generate synthetic dataset
df = generate_synthetic_data(10000)

# Prepare features and labels
X = df.drop(columns='Growth_Milestone')
y = df['Growth_Milestone']

# Convert categorical 'Soil_Type' to numerical
X = pd.get_dummies(X, columns=['Soil_Type'], drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model performance on the test data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Seedling', 'Early Growth', 'Mature'], yticklabels=['Seedling', 'Early Growth', 'Mature'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Learning Curves (Training vs Cross-validation scores)
train_sizes = [1, 2, 3, 4, 5]  # Example train sizes, modify as needed
train_scores = [model.score(X_train_scaled[:i], y_train[:i]) for i in train_sizes]
cv_scores = [model.score(X_test_scaled[:i], y_test[:i]) for i in train_sizes]

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, label='Training score', color='blue')
plt.plot(train_sizes, cv_scores, label='Cross-validation score', color='orange')
plt.fill_between(train_sizes, train_scores, cv_scores, color='lightgray', alpha=0.5)
plt.title('Learning Curves')
plt.xlabel('Number of Training Samples')
plt.ylabel('Score')
plt.legend()
plt.show()

# Save the model
import joblib
joblib.dump(model, 'random_forest_model.pkl')
