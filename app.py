import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #ff4b4b;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
}

.result-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load files
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Title
st.markdown('<p class="title">❤️ Heart Disease Prediction</p>',
            unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Developed by Samih | AI-powered Heart Risk Assessment</p>',
    unsafe_allow_html=True
)

# Sidebar Inputs
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Gender", ["M", "F"])
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.sidebar.number_input(
    "Resting Blood Pressure",
    80, 250, 120
)

cholesterol = st.sidebar.number_input(
    "Cholesterol",
    100, 700, 200
)

fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar >120",
    [0, 1]
)

resting_ecg = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.sidebar.slider(
    "Maximum Heart Rate",
    60, 220, 150
)

exercise_angina = st.sidebar.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

oldpeak = st.sidebar.slider(
    "Oldpeak",
    0.0, 6.0, 1.0
)

st_slope = st.sidebar.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# Main Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Age", age)

with col2:
    st.metric("Blood Pressure", resting_bp)

with col3:
    st.metric("Cholesterol", cholesterol)

# Predict Button
if st.button("🔍 Predict Heart Disease Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    # Probability (if available)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)[0][1]
    else:
        probability = 0.5

    st.subheader("Prediction Result")

    st.progress(int(probability * 100))

    st.metric(
        label="Risk Score",
        value=f"{probability*100:.1f}%"
    )

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.warning("""
        Recommendations:
        - Consult a cardiologist
        - Reduce cholesterol intake
        - Exercise regularly
        - Monitor blood pressure
        """)
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.info("""
        Great! Continue:
        - Healthy diet
        - Regular exercise
        - Annual health checkups
        """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Scikit-Learn and Python")