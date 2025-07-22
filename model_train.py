# model_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the Excel dataset
df = pd.read_excel("heart_failure_clinical_records_dataset(1).csv")

# Features and target
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")
