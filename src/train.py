import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Resolve paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student.csv")


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=";")


# -----------------------------
# Features & target
# -----------------------------
features = ["studytime", "absences", "G1", "G2"]
target = "G3"

X = df[features]
y = df[target]


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Helper function to evaluate models
# -----------------------------
def evaluate(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n===== {name} =====")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R2 Score: {r2:.3f}")


# -----------------------------
# Models
# -----------------------------
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# -----------------------------
# Evaluate all models
# -----------------------------
evaluate(lr, "Linear Regression")
evaluate(dt, "Decision Tree Regressor")
evaluate(rf, "Random Forest Regressor")
