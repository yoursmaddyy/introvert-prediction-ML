import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained model
introvert_model = load('introvert_model.pkl')

st.title("Introvert Predictor 🧠")

st.write("Enter the following details to predict if a person is an Introvert or not:")

# Input widgets
time_spent_alone = st.number_input("Time spent alone (hours per day)", min_value=0, max_value=24, value=3)

# Binary inputs with Yes/No
stage_fear = st.radio("Do you have stage fear?", ["No", "Yes"])
drained_after_socializing = st.radio("Do you feel drained after socializing?", ["No", "Yes"])

# Convert Yes/No to 0/1
stage_fear = 1 if stage_fear == "Yes" else 0
drained_after_socializing = 1 if drained_after_socializing == "Yes" else 0

# Numeric inputs
social_event_attendance = st.slider("Social event attendance per month", min_value=0, max_value=30, value=2)
going_outside = st.slider("Going outside per week", min_value=0, max_value=7, value=3)
friends_circle_size = st.slider("Friends circle size", min_value=0, max_value=50, value=5)
post_frequency = st.slider("Post frequency (per week)", min_value=0, max_value=50, value=5)

# Prepare input DataFrame
input_data = pd.DataFrame([[
    time_spent_alone,
    stage_fear,
    social_event_attendance,
    going_outside,
    drained_after_socializing,
    friends_circle_size,
    post_frequency
]], columns=[
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency"
])

# Predict button
if st.button("Predict"):
    prediction = introvert_model.predict(input_data)[0]
    prob = introvert_model.predict_proba(input_data)[0]
    
    st.write("### Prediction:")
    if prediction:
        st.success("This person is likely an **Introvert**")
    else:
        st.success("This person is likely **Not an Introvert**")
    
    st.write("### Prediction Probabilities:")
    st.write(f"Probability of Not Introvert: {prob[0]:.2f}")
    st.write(f"Probability of Introvert    : {prob[1]:.2f}")
