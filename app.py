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
    age = st.number_input("Age", min_value=0, max_value=70, value=0, step=1)
    marital_status = st.selectbox("Marital Status", options=["Yes", "No"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    bmi = st.number_input("BMI", min_value=2.0, max_value=4.0, value=2.0, step=0.1)
    snoring_rate = st.number_input("Snoring Rate", min_value=-1.0, max_value=8.0, value=-1.0, step=0.1)
    respiration_rate = st.number_input("Respiration Rate", min_value=-1.0, max_value=3.0, value=-1.0, step=0.1)
    body_temperature = st.number_input("Body Temperature", min_value=80.0, max_value=100.0, value=80.0, step=0.1)
    limb_movement = st.number_input("Limb Movement", min_value=-2.0, max_value=4.0, value=-2.0, step=0.1)
    blood_oxygen = st.number_input("Blood Oxygen", min_value=79.0, max_value=100.0, value=79.0, step=0.1)
    eye_movement = st.number_input("Eye Movement", min_value=0.0, max_value=8.0, value=0.0, step=0.1)
    sleeping_hours = st.number_input("Sleeping Hours", min_value=0.0, max_value=9.0, value=0.0, step=0.1)
    heart_rate = st.number_input("Heart Rate", min_value=-1.0, max_value=3.0, value=-1.0, step=0.1)

    marital_status = 1 if marital_status == "Yes" else 0
    gender = 1 if gender == "Male" else 0

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
    prediction = model.predict(user_input)[0]
    
    # Define stress levels and corresponding descriptions
    stress_descriptions = {
        0: "No Stress",
        1: "Low Stress",
        2: "Moderate Stress",
        3: "High Stress",
        4: "Max Stress"
    }
    stress_level = stress_descriptions.get(prediction, "Unknown")
    
    # Define colors for the stress levels
    colors = ['#d0f0c0', '#b0e57c', '#f2b700', '#f77f00', '#d62839']
    
    # Create a half-circular gauge chart
    fig = go.Figure()

    # Add the curved bar sections
    for i in range(5):
        fig.add_trace(go.Barpolar(
            r=[1] * (i + 1) + [0],
            theta=[(i * 36 + 18)] * (i + 1) + [0],  # Center the bars in the half-circle
            marker_color=colors[i],
            showlegend=False
        ))

    # Add a pointer (arrow) to indicate the stress level
    fig.add_trace(go.Scatterpolar(
        r=[1],
        theta=[prediction * 36 + 18],  # Position the arrow based on the prediction
        mode='markers+text',
        marker=dict(size=15, color='black', symbol='arrow-bar-up'),
        text=[f'<b>{stress_level}</b>'],
        textposition='top center',
        showlegend=False
    ))

    # Add annotation for better visualization
    fig.add_annotation(
        x=prediction * 36 + 18,
        y=1.1,
        text=f"<b>{stress_level}</b>",
        showarrow=True,
        font=dict(size=14, color="black"),
        align="center",
        bgcolor="white",
        bordercolor=colors[prediction],
        borderwidth=2,
        borderpad=4
    )

    # Update layout for a half-circle gauge design
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 1]
            ),
            angularaxis=dict(
                tickvals=[i * 36 + 18 for i in range(5)],
                ticktext=["No Stress", "Low Stress", "Moderate Stress", "High Stress", "Max Stress"],
                showgrid=False,
                showticklabels=True
            )
        ),
        showlegend=False,
        height=400,
        width=600,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Display the gauge chart in Streamlit
    st.plotly_chart(fig)

    # Show the predicted stress level below the chart
    st.subheader(f"Predicted Stress Level: {stress_level} (Level {prediction})")
