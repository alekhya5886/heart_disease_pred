import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
num_samples = 1000

data = {
    'Gender': np.random.choice([0, 1], size=num_samples),
    'Age': np.random.randint(25, 65, size=num_samples),
    'FamilyHistory': np.random.choice([0, 1], size=num_samples),
    'PastHeartIssues': np.random.choice([0, 1], size=num_samples),
    'PhysicalActivity': np.random.choice([0, 1], size=num_samples),
    'Diet': np.random.choice([0, 1], size=num_samples),
    'Smoking': np.random.choice([0, 1], size=num_samples),
    'AlcoholConsumption': np.random.choice([0, 1], size=num_samples),
    'SleepPatterns': np.random.choice([0, 1], size=num_samples),
    'HighBloodPressure': np.random.choice([0, 1], size=num_samples),
    'Diabetic': np.random.choice([0, 1], size=num_samples),
}

df = pd.DataFrame(data)
df.to_csv('heart_disease_dataset.csv', index=False)
