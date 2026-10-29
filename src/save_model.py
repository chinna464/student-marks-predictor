import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Load data
df = pd.read_csv(DATA_PATH, sep=";")

features = ["studytime", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]

# Train model
model = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X, y)

# Save model
joblib.dump(model, MODEL_PATH)

print("Model saved successfully as model.pkl")
