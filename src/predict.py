import os
import joblib
import numpy as np

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_marks(studytime, absences, g1, g2):
    """
    Predict final marks (G3) given input features
    """
    input_data = np.array([[studytime, absences, g1, g2]])
    prediction = model.predict(input_data)
    return prediction[0]


# Test prediction (ONLY for backend testing)
if __name__ == "__main__":
    result = predict_marks(
        studytime=2,   # weekly study time
        absences=5,    # number of absences
        g1=14,         # first internal
        g2=15          # second internal
    )

    print(f"Predicted Final Marks: {result:.2f}")
