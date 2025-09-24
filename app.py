import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained model
introvert_model = load('introvert_model.pkl')

st.title("Introvert Predictor 🧠")

st.write("Enter the following details to predict if a person is an Introvert or not:")

# Input widgets
time_spent_alone = st.number_input("Time spent alone (hours per day)", min_value=0, max_value=10, value=3)
stage_fear = st.slider("Stage fear (0-5)", min_value=0, max_value=1, value=0)
social_event_attendance = st.slider("Social event attendance (0-10)", min_value=0, max_value=10, value=0)
going_outside = st.slider("Going outside (0-10)", min_value=0, max_value=10, value=6)
drained_after_socializing = st.slider("Drained after socializing (0-10)", min_value=0, max_value=1, value=0)
friends_circle_size = st.slider("Friends circle size (0-20)", min_value=0, max_value=20, value=5)
post_frequency = st.slider("Post frequency (0-20)", min_value=0, max_value=20, value=5)

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
