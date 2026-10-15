import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Load the trained model
MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
MODEL_PATH = MODEL_DIR / "hr_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

model_columns = model.get_booster().feature_names

# Page setup
st.set_page_config(page_title="Team NEXUS HR Portal", layout="centered")
st.title("📊 HR Attrition Predictor")
st.markdown("Analyze employee flight risk in real time.")
st.divider()

# Employee inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    income = st.number_input(
        "Monthly Income (Rs)", min_value=1, max_value=500000, value=5000
    )
    hours_worked = st.number_input(
        "Average Hours/Week", min_value=19, max_value=80, value=40
    )
    distance = st.number_input(
        "Distance From Home", min_value=1, max_value=50, value=5
    )
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"])

with col2:
    work_life = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
    job_sat = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    department = st.selectbox(
        "Department", ["Finance", "HR", "Marketing", "Sales", "IT"]
    )
    job_role = st.selectbox(
        "Job Role", ["Manager", "Assistant", "Analyst", "Executive"]
    )
    years = st.number_input(
        "Years at Company", min_value=0, max_value=40, value=3
    )

# Default values for model features without input controls
baseline_employee = {
    "Job_Level": 3,
    "Hourly_Rate": 65,
    "Years_in_Current_Role": 3,
    "Years_Since_Last_Promotion": 1,
    "Performance_Rating": 3,
    "Training_Hours_Last_Year": 3,
    "Project_Count": 4,
    "Absenteeism": 2,
    "Work_Environment_Satisfaction": 3,
    "Relationship_with_Manager": 3,
    "Job_Involvement": 3,
    "Number_of_Companies_Worked": 2,
}

# Combine defaults with entered values
input_data = {
    **baseline_employee,
    "Age": age,
    "Monthly_Income": income,
    "Average_Hours_Worked_Per_Week": hours_worked,
    "Distance_From_Home": distance,
    "Work_Life_Balance": work_life,
    "Job_Satisfaction": job_sat,
    "Years_at_Company": years,
    "Overtime": 1 if overtime == "Yes" else 0,
    f"Department_{department}": 1,
    f"Job_Role_{job_role}": 1,
}

# Build the model input
input_df = pd.DataFrame([input_data]).reindex(
    columns=model_columns,
    fill_value=0,
)

if st.button("Analyze Risk", type="primary"):
    classes = list(model.classes_)

    if 1 not in classes:
        st.error(f"Expected attrition label 1; model classes are {classes}")
    else:
        positive_index = classes.index(1)
        risk = float(model.predict_proba(input_df)[0][positive_index]) * 100

        st.divider()
        st.metric("Predicted turnover probability", f"{risk:.1f}%")