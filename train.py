import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("heart_disease_uci.csv")

# Drop unneeded columns (only if present)
df = df.drop(columns=['id', 'dataset'], errors='ignore')

# Encode categorical columns
label_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
df[label_cols] = df[label_cols].apply(LabelEncoder().fit_transform)

# Handle missing values
df = df.dropna()

# Features and target
X = df.drop('num', axis=1)
y = (df['num'] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as heart_model.pkl")