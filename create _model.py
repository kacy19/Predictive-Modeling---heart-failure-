# Run this script to create the model.pkl file
# Make sure you have your heart failure dataset CSV file

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Sample data creation (replace with your actual dataset loading)
# If you have the actual dataset, replace the sample data creation with:
# df = pd.read_csv('your_heart_failure_dataset.csv')

# Create sample dataset (replace this with actual data loading)
np.random.seed(42)
n_samples = 299

# Generate sample data similar to heart failure dataset
data = {
    'age': np.random.randint(40, 95, n_samples),
    'anaemia': np.random.randint(0, 2, n_samples),
    'creatinine_phosphokinase': np.random.randint(23, 7861, n_samples),
    'diabetes': np.random.randint(0, 2, n_samples),
    'ejection_fraction': np.random.randint(14, 80, n_samples),
    'high_blood_pressure': np.random.randint(0, 2, n_samples),
    'platelets': np.random.uniform(25100, 850000, n_samples),
    'serum_creatinine': np.random.uniform(0.5, 9.4, n_samples),
    'serum_sodium': np.random.randint(113, 148, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'smoking': np.random.randint(0, 2, n_samples),
    'time': np.random.randint(4, 285, n_samples)
}

# Create target variable based on some logic (replace with actual target)
death_event = []
for i in range(n_samples):
    # Simple logic for demo - replace with actual target from your dataset
    risk_score = (data['age'][i] > 65) + (data['ejection_fraction'][i] < 30) + \
                 (data['serum_creatinine'][i] > 2.0) + (data['diabetes'][i]) + \
                 (data['high_blood_pressure'][i])
    death_event.append(1 if risk_score >= 3 else 0)

data['DEATH_EVENT'] = death_event
df = pd.DataFrame(data)

print("Sample dataset created with shape:", df.shape)
print("Target distribution:", df['DEATH_EVENT'].value_counts())

# Prepare features and target
X = df.drop(['DEATH_EVENT'], axis=1)
y = df['DEATH_EVENT']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

# Test accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save model
model_data = {
    'model': model,
    'scaler': scaler,
    'feature_names': X.columns.tolist()
}

with open('heart_failure_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully as 'heart_failure_model.pkl'")
print("Feature names:", X.columns.tolist())
