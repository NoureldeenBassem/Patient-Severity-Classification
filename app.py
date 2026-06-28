import streamlit as st
import pandas as pd
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Severity Classifier",
    page_icon="🏥",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("deployment_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

severity_labels = {0: "Healthy", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Critical"}

severity_colors = {
    "Healthy":  ("#1a7a1a", "#d4edda"),   # dark green text, light green bg
    "Mild":     ("#1a5276", "#d6eaf8"),   # dark blue text, light blue bg
    "Moderate": ("#7d6608", "#fef9e7"),   # dark yellow text, light yellow bg
    "Severe":   ("#a04000", "#fdebd0"),   # dark orange text, light orange bg
    "Critical": ("#7b241c", "#fadbd8"),   # dark red text, light red bg
}

severity_descriptions = {
    "Healthy":  "The patient shows no significant clinical concerns.",
    "Mild":     "The patient has minor symptoms that require monitoring.",
    "Moderate": "The patient has notable symptoms and needs medical attention.",
    "Severe":   "The patient is in a serious condition and requires urgent care.",
    "Critical": "The patient is in a critical condition and needs immediate intervention.",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Patient Severity Classifier")
st.markdown("Enter the patient's clinical measurements below to predict their severity level.")
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics**")
    age    = st.number_input("Age (years)",          min_value=1,   max_value=120, value=45)
    gender = st.selectbox("Gender",                  ["Male", "Female"])
    bmi    = st.number_input("BMI (kg/m²)",          min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    st.markdown("**Lifestyle**")
    smoking_status = st.selectbox("Smoking Status",  ["No", "Yes", "Former"])
    diabetes       = st.selectbox("Diabetes",        ["No", "Yes"])

with col2:
    st.markdown("**Vital Signs**")
    bp_sys  = st.number_input("Blood Pressure — Systolic (mmHg)",  min_value=60,  max_value=250, value=120)
    bp_dia  = st.number_input("Blood Pressure — Diastolic (mmHg)", min_value=40,  max_value=150, value=80)
    heart_rate = st.number_input("Heart Rate (BPM)",               min_value=30,  max_value=200, value=75)

    st.markdown("**Lab Results**")
    glucose     = st.number_input("Glucose Level (mg/dL)",   min_value=50.0,  max_value=500.0, value=100.0, step=0.1)
    cholesterol = st.number_input("Cholesterol (mg/dL)",     min_value=50.0,  max_value=400.0, value=180.0, step=0.1)
    wbc         = st.number_input("WBC Count (×10³/μL)",     min_value=1.0,   max_value=30.0,  value=7.0,   step=0.1)
    hemoglobin  = st.number_input("Hemoglobin (g/dL)",       min_value=4.0,   max_value=20.0,  value=14.0,  step=0.1)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Severity", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([{
        "age":                      float(age),
        "bmi":                      float(bmi),
        "blood_pressure_systolic":  float(bp_sys),
        "blood_pressure_diastolic": float(bp_dia),
        "glucose_level":            float(glucose),
        "cholesterol":              float(cholesterol),
        "heart_rate":               float(heart_rate),
        "wbc_count":                float(wbc),
        "hemoglobin":               float(hemoglobin),
        "gender":                   gender,
        "smoking_status":           smoking_status,
        "diabetes":                 diabetes,
    }])

    prediction   = model.predict(input_df)[0]
    label        = severity_labels[prediction]
    text_color, bg_color = severity_colors[label]
    description  = severity_descriptions[label]

    st.markdown("### Prediction Result")
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border-left: 6px solid {text_color};
            padding: 20px 24px;
            border-radius: 8px;
            margin-bottom: 12px;
        ">
            <p style="margin:0; font-size:14px; color:{text_color}; font-weight:600; text-transform:uppercase; letter-spacing:1px;">
                Predicted Severity
            </p>
            <p style="margin:4px 0 8px 0; font-size:32px; font-weight:800; color:{text_color};">
                {label}
            </p>
            <p style="margin:0; font-size:15px; color:{text_color};">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Input Summary")
    summary_df = pd.DataFrame({
        "Feature": [
            "Age", "Gender", "BMI", "Smoking Status", "Diabetes",
            "BP Systolic", "BP Diastolic", "Heart Rate",
            "Glucose", "Cholesterol", "WBC Count", "Hemoglobin"
        ],
        "Value": [
            age, gender, bmi, smoking_status, diabetes,
            bp_sys, bp_dia, heart_rate,
            glucose, cholesterol, wbc, hemoglobin
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Patient Severity Classifier · AI & Data Science Case Study · Noureldeen Bassem")
