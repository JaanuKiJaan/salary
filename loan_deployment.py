# -*- coding: utf-8 -*-
"""loan_deployment"""

import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model_and_encoder():
    model_path = "salary_prediction_model.pkl"
    encoder_path = "label_encoder.pkl"
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)

        # Ensure encoder is dictionary
        if not isinstance(encoder, dict):
            st.error("Encoder file is not saved correctly. It must be a dictionary.")
            st.stop()

        return model, encoder

    except FileNotFoundError:
        st.error(
            f"Error: Model or encoder file not found. "
            f"Make sure '{model_path}' and '{encoder_path}' are in the correct directory."
        )
        st.stop()

    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        st.stop()


model, encoder = load_model_and_encoder()

st.title("Salary Prediction App")
st.write("Enter the details below to predict the salary:")

# Ensure required keys exist
required_keys = ["Gender", "Education Level", "Job Title"]
for key in required_keys:
    if key not in encoder:
        st.error(f"Encoder is missing required key: {key}")
        st.stop()

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", encoder["Gender"].classes_)
education = st.selectbox("Education Level", encoder["Education Level"].classes_)
job_title = st.selectbox("Job Title", encoder["Job Title"].classes_)
years_of_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [years_of_exp]
})

if st.button("Predict Salary"):

    df_encoded = input_data.copy()

    for col_name, label_encoder in encoder.items():
        if col_name in df_encoded.columns:
            try:
                df_encoded[col_name] = label_encoder.transform(
                    df_encoded[col_name]
                )
            except ValueError as e:
                st.error(f"Error encoding '{col_name}': {e}")
                st.stop()

    try:
        prediction = model.predict(df_encoded)
        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
