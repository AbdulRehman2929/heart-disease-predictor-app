import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="CardioCheck | Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Custom styling (crimson / slate theme)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1050px; }
    .hero {
        background: linear-gradient(120deg, #7f1d1d 0%, #be123c 55%, #fb7185 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: #fff;
        box-shadow: 0 8px 24px rgba(190, 18, 60, 0.25); margin-bottom: 1.6rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: .3px; color: #fff; }
    .hero p  { margin: .5rem 0 0; font-size: 1rem; opacity: .92; color: #fff; }
    .section-label {
        font-weight: 700; font-size: 1.05rem; color: #be123c;
        border-left: 5px solid #be123c; padding-left: .6rem; margin: .4rem 0 1rem;
    }
    .result-card {
        padding: 1.4rem 1.6rem; border-radius: 16px; margin-top: 1rem;
        font-size: 1.25rem; font-weight: 700; text-align: center;
    }
    .result-risk { background: #fff1f2; border: 2px solid #e11d48; color: #9f1239; }
    .result-safe { background: #ecfdf5; border: 2px solid #10b981; color: #065f46; }
    .result-card small { display: block; font-weight: 400; font-size: .9rem; margin-top: .4rem; }
    .stButton > button {
        background: #be123c; color: #fff; border: none; border-radius: 10px;
        padding: .6rem 1.4rem; font-weight: 600; width: 100%;
    }
    .stButton > button:hover { background: #9f1239; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------
@st.cache_resource
def fetch_trained_model():
    with open("heart_svc_trained_model.pkl", "rb") as model_file:
        return pickle.load(model_file)


svc_heart_model = fetch_trained_model()

# ---------------------------------------------------------------
# Category lists + label encoders (same classes as used in training)
# ---------------------------------------------------------------
cp_choices = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
exang_choices = ["Yes", "No"]
thal_choices = ["Normal", "Fixed Defect", "Reversible Defect"]
ca_choices = ["0 Vessels", "1 Vessel", "2 Vessels", "3 Vessels"]

cp_encoder = LabelEncoder().fit(np.ravel(cp_choices))
exang_encoder = LabelEncoder().fit(np.ravel(exang_choices))
thal_encoder = LabelEncoder().fit(np.ravel(thal_choices))
ca_encoder = LabelEncoder().fit(np.ravel(ca_choices))

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("🫀 About CardioCheck")
    st.write(
        "A small machine-learning demo that estimates the likelihood of heart "
        "disease from four clinical indicators."
    )
    st.markdown("**Model:** Support Vector Classifier")
    st.markdown("**Dataset:** UCI Heart Disease (Cleveland)")
    st.markdown("**Inputs used:** 4 attributes")
    st.divider()
    st.caption("Course: AIC354 - Machine Learning Fundamentals")
    st.caption("COMSATS University Islamabad, Lahore Campus")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🫀 CardioCheck</h1>
        <p>Heart Disease Prediction System &mdash; fill in the four clinical
        indicators and let the trained model assess the case.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
st.markdown('<div class="section-label">Clinical Indicators</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    pick_cp = st.selectbox("Chest Pain Type", cp_choices, help="Type of chest pain reported by the patient.")
    pick_thal = st.selectbox("Thalassemia Test Result", thal_choices, help="Result of the thallium stress test.")

with col_right:
    pick_exang = st.selectbox("Exercise Induced Angina", exang_choices, help="Does exercise trigger chest pain?")
    pick_ca = st.selectbox("Major Vessels Colored by Fluoroscopy", ca_choices, help="Number of major vessels with blockage.")

st.write("")
run_prediction = st.button("🔍 Analyse Patient")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if run_prediction:
    patient_row = pd.DataFrame({
        "ChestPain": [pick_cp],
        "ExerciseAngina": [pick_exang],
        "Thal": [pick_thal],
        "Vessels": [pick_ca],
    })

    encoded_row = patient_row.copy()
    encoded_row["ChestPain"] = cp_encoder.transform(patient_row["ChestPain"])
    encoded_row["ExerciseAngina"] = exang_encoder.transform(patient_row["ExerciseAngina"])
    encoded_row["Thal"] = thal_encoder.transform(patient_row["Thal"])
    encoded_row["Vessels"] = ca_encoder.transform(patient_row["Vessels"])

    outcome = svc_heart_model.predict(encoded_row)[0]

    if outcome == 1:
        st.markdown(
            '<div class="result-card result-risk">⚠️ HEART DISEASE DETECTED'
            "<small>The model flags this profile as high risk. Please consult a cardiologist.</small></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-card result-safe">✅ NO HEART DISEASE'
            "<small>The model does not flag this profile as at risk.</small></div>",
            unsafe_allow_html=True,
        )

    with st.expander("View submitted patient data"):
        st.dataframe(patient_row, hide_index=True)
        st.caption("Numeric codes sent to the model:")
        st.dataframe(encoded_row, hide_index=True)

st.write("")
st.caption("Learning project (AIC354, COMSATS Lahore). This is not a medical diagnosis tool.")
