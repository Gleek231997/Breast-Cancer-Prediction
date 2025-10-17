import streamlit as st
import pickle
import numpy as np

# ------------------------------
# Load Model
# ------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="centered"
)

st.title("🎗️ Breast Cancer Prediction")
st.markdown(
    "Predict whether a tumor is **malignant** or **benign** using a pre-trained ML model."
)
st.write(
    "Model trained on selected features of the [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)."
)

# ------------------------------
# Sidebar Input
# ------------------------------
st.sidebar.header("Input Features")
def user_input_features():
    mean_radius = st.sidebar.slider("Mean Radius", 6.0, 30.0, 14.0)
    mean_texture = st.sidebar.slider("Mean Texture", 9.0, 40.0, 20.0)
    mean_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, 80.0)
    mean_area = st.sidebar.slider("Mean Area", 100.0, 2500.0, 500.0)
    mean_smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.2, 0.1)
    features = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
    return np.array(features).reshape(1, -1)

features = user_input_features()

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]

    label = "Benign 🟢" if prediction == 1 else "Malignant 🔴"
    st.subheader("Prediction")
    st.markdown(
        f"<h2 style='color: {'green' if prediction==1 else 'red'};'>{label}</h2>",
        unsafe_allow_html=True
    )

    st.subheader("Prediction Probability")
    st.write(f"Benign: {prediction_proba[1]*100:.2f}% | Malignant: {prediction_proba[0]*100:.2f}%")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    "🧩 **Project by [Glory Ekbote](https://github.com/Gleek231997)**  \n"
    "⚙️ Built with Streamlit & Scikit-learn"
)

