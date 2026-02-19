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

        if not isinstance(encoder, dict):
            st.error("Encoder file must be a dictionary of LabelEncoders.")
            st.stop()

        return model, encoder

    except FileNotFoundError:
        st.error(
            f"Model or encoder file not found. "
            f"Ensure '{model_path}' and '{encoder_path}' exist."
        )
        st.stop()

    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        st.stop()


model, encoder = load_model_and_encoder()

st.title("Salary Prediction App")
st.write("Enter the details below to predict the salary:")


required_columns = ["Gender", "Education Level", "Job Title"]

for col in required_columns:
    if col not in encoder:
        st.error(f"Encoder missing required column: '{col}'")
        st.stop()


age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox(
    "Gender",
    list(encoder["Gender"].classes_)
)

education = st.selectbox(
    "Education Level",
    list(encoder["Education Level"].classes_)
)

job_title = st.selectbox(
    "Job Title",
    list(encoder["Job Title"].classes_)
)

years_of_exp = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=5
)


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
                    df_encoded[col_name].astype(str)
                )
            except Exception as e:
                st.error(f"Encoding error in '{col_name}': {e}")
                st.stop()

    try:
        prediction = model.predict(df_encoded)

        # Ensure prediction works even if array format changes
        predicted_salary = float(prediction[0])

        st.success(f"Predicted Salary: ${predicted_salary:,.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
