import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("🏥 End-to-End Diabetes Risk Prediction Pipeline")
st.markdown("""
Built with **Python (Pandas, NumPy, Scikit-learn)**. This dashboard demonstrates an end-to-end machine learning pipeline, featuring EDA, data cleaning, and feature engineering to surface actionable health insights.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Input Features")
def user_input_features():
    age = st.sidebar.slider("Age", 21, 90, 45)
    bmi = st.sidebar.slider("BMI", 15.0, 50.0, 28.5)
    glucose = st.sidebar.slider("Fasting Glucose (mg/dL)", 70, 200, 110)
    blood_pressure = st.sidebar.slider("Blood Pressure (Diastolic)", 50, 120, 80)
    family_history = st.sidebar.selectbox("Family History of Diabetes", (0, 1))
    
    data = {'Age': age,
            'BMI': bmi,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'FamilyHistory': family_history}
    
    # Correctly defining the first row of data for Pandas
    features = pd.DataFrame(data, index=)
    return features

input_df = user_input_features()

# --- Mock Model Training (Recreating your Scikit-learn pipeline) ---
@st.cache_resource
def build_model():
    # Generating mock large-scale patient dataset for demonstration
    np.random.seed(42)
    X_mock = pd.DataFrame({
        'Age': np.random.randint(21, 90, 1000),
        'BMI': np.random.uniform(15.0, 50.0, 1000),
        'Glucose': np.random.randint(70, 200, 1000),
        'BloodPressure': np.random.randint(50, 120, 1000),
        'FamilyHistory': np.random.randint(0, 2, 1000)
    })
    # Mock target variable based on some logical rules
    y_mock = ((X_mock['Glucose'] > 125) | ((X_mock['BMI'] > 30) & (X_mock['Age'] > 50))).astype(int)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_mock, y_mock)
    return clf, X_mock

model, X_train = build_model()

# --- Dashboard Display ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Preprocessed Patient Data")
    st.write("Simulating validated data structures typically cleaned via Power Query & Pivot Tables:")
    st.dataframe(input_df)

    st.subheader("2. Model Prediction")
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Correctly extracting the specific values from the prediction array
    if prediction == 1:
        st.error(f"**High Risk** detected. (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"**Low Risk** detected. (Probability: {prediction_proba:.2f})")

with col2:
    st.subheader("3. Feature Engineering & Importance")
    st.write("Top predictive features driving this model's business logic:")
    
    # Extract feature importance from the model
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.bar_chart(importance.set_index('Feature'))

st.markdown("---")
st.subheader("Actionable Insights for Business Stakeholders")
st.markdown("""
* **Targeted Interventions:** Patients with elevated Glucose combined with high BMI warrant immediate lifestyle intervention programs.
* **Resource Allocation:** Predictive modeling allows clinics to allocate screening resources to high-risk demographic clusters efficiently.
""")
