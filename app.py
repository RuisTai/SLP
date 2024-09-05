import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

# Streamlit app title
st.title("Stress Prediction System")

# Introduction text
st.write("Enter the features below to predict the stress level (0 to 4):")

# Function to get user input
def get_user_input():
    age = st.number_input("Age", min_value=0, max_value=70, value=0, step=1)  # Int values
    marital_status = st.selectbox("Marital Status", options=["Yes", "No"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    bmi = st.number_input("BMI", min_value=2.0, max_value=4.0, value=2.0, step=0.1)  # Float values
    snoring_rate = st.number_input("Snoring Rate", min_value=-1.0, max_value=8.0, value=-1.0, step=0.1)  # Float values
    respiration_rate = st.number_input("Respiration Rate", min_value=-1.0, max_value=3.0, value=-1.0, step=0.1)  # Float values
    body_temperature = st.number_input("Body Temperature", min_value=80.0, max_value=100.0, value=80.0, step=0.1)  # Float values
    limb_movement = st.number_input("Limb Movement", min_value=-2.0, max_value=4.0, value=-2.0, step=0.1)  # Float values
    blood_oxygen = st.number_input("Blood Oxygen", min_value=79.0, max_value=100.0, value=79.0, step=0.1)  # Float values
    eye_movement = st.number_input("Eye Movement", min_value=0.0, max_value=8.0, value=0.0, step=0.1)  # Float values
    sleeping_hours = st.number_input("Sleeping Hours", min_value=0.0, max_value=9.0, value=0.0, step=0.1)  # Float values
    heart_rate = st.number_input("Heart Rate", min_value=-1.0, max_value=3.0, value=-1.0, step=0.1)  # Float values

    # Convert marital status and gender to numerical values
    marital_status = 1 if marital_status == "Yes" else 0  # Update to match "Yes"/"No"
    gender = 1 if gender == "Male" else 0

    # Create an array of user inputs
    input_data = np.array([
        age, marital_status, gender, bmi, snoring_rate, respiration_rate,
        body_temperature, limb_movement, blood_oxygen, eye_movement,
        sleeping_hours, heart_rate
    ]).reshape(1, -1)

    return input_data

# Get user input
user_input = get_user_input()

# Predict button
if st.button("Predict Stress Level"):
    # Predict the stress level
    prediction = model.predict(user_input)
    
    # Map the prediction to stress level description
    stress_descriptions = {
        0: "No Stress",
        1: "Low Stress",
        2: "Moderate Stress",
        3: "High Stress",
        4: "Max Stress"
    }
    stress_level = stress_descriptions.get(prediction[0], "Unknown")

    # Display the prediction
    st.subheader(f"Predicted Stress Level: {stress_level} (Level {prediction[0]})")

    # Create a gauge chart with proper colors and an arrow indicator
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction[0],
        title={'text': "Stress Level"},
        gauge={
            'axis': {'range': [0, 4], 'tickvals': [0, 1, 2, 3, 4], 'ticktext': ['No', 'Low', 'Moderate', 'High', 'Max']},
            'bar': {'color': "white"},  # No center bar
            'steps': [
                {'range': [0, 1], 'color': "lime"},      # No Stress
                {'range': [1, 2], 'color': "green"},     # Low Stress
                {'range': [2, 3], 'color': "yellow"},    # Moderate Stress
                {'range': [3, 4], 'color': "orange"},    # High Stress
                {'range': [4, 5], 'color': "red"}        # Max Stress
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'thickness': 0.75,
                'value': prediction[0]
            }
        }
    ))

    # Render the gauge chart in Streamlit
    st.plotly_chart(fig)

