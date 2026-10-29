import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Student Marks Predictor",
    layout="centered"
)

st.title("Student Marks Predictor")
st.caption("Predict final examination marks using academic performance data")

st.divider()

st.subheader("Enter Student Details")

studytime = st.slider(
    "Study Time per Week",
    1, 4, 2,
    help="1 = Very Low, 2 = Low, 3 = Moderate, 4 = High"
)

absences = st.slider(
    "Number of Classes Missed",
    0, 30, 5
)

g1 = st.slider(
    "First Internal Marks (G1)",
    0, 20, 12
)

g2 = st.slider(
    "Second Internal Marks (G2)",
    0, 20, 14
)

st.divider()

if st.button("Predict Final Marks"):
    X = np.array([[studytime, absences, g1, g2]])
    prediction = round(model.predict(X)[0], 2)

    st.success("Prediction Successful")

    st.metric(
        label="Predicted Final Marks (G3)",
        value=f"{prediction} / 20"
    )

    if g2 < g1:
        st.info("Second internal marks are lower than the first internal marks.")
