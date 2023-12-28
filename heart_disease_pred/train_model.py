import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('heart_disease_dataset.csv')

# Features and target variable
selected_features = ['Gender', 'Age', 'FamilyHistory', 'PastHeartIssues', 'PhysicalActivity', 'Diet', 'Smoking', 'AlcoholConsumption', 'SleepPatterns', 'HighBloodPressure', 'Diabetic']
X = df[selected_features]
y = df['Diabetic']

# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model to a file
pickle.dump(model, open('heart_disease_model.pkl', 'wb'))
