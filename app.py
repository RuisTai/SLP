import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

# Streamlit app title
st.title("Stress Prediction System")

# Sidebar for input section
with st.sidebar:
    st.write("Enter the features below to predict the stress level from 0 to 4:")

# Function to get user input
def get_user_input():
    age = st.number_input("Age (Enter input : 18~80)", min_value=18, max_value=80, value=22, step=1)
    marital_status = st.selectbox("Marital Status", options=["Yes", "No"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    bmi = st.number_input("BMI (Enter input : 18.0~40.0)", min_value=18.0, max_value=40.0, value=25.0, step=0.1)
    snoring_rate = st.number_input("Snoring Rate (Enter input : 0~50)", min_value=-1.0, max_value=50.0, value=5.0, step=0.1)
    respiration_rate = st.number_input("Respiration Rate  (Enter input : 0~50)", min_value=-1.0, max_value=50.0, value=15.0, step=0.1)
    body_temperature = st.number_input("Body Temperature °F (Enter input : 60~110)", min_value=60.0, max_value=110.0, value=90.0, step=0.1)
    limb_movement = st.number_input("Limb Movement (Enter input : 0~35)", min_value=-1.0, max_value=35.0, value=3.0, step=0.1)
    blood_oxygen = st.number_input("Blood Oxygen (Enter input : 60~110)", min_value=60.0, max_value=110.0, value=80.0, step=0.1)
    eye_movement = st.number_input("Eye Movement  (Enter input : 0~35)", min_value=-1.0, max_value=35.0, value=20.0, step=0.1)
    sleeping_hours = st.number_input("Sleeping Hours (Enter input : 0~24)", min_value=-1.0, max_value=24.0, value=8.0, step=0.1)
    heart_rate = st.number_input("Heart Rate (Enter input : 30~100)", min_value=30.0, max_value=100.0, value=70.0, step=0.1)

    marital_status = 1 if marital_status == "Yes" else 0
    gender = 1 if gender == "Male" else 0

    input_data = np.array([
        age, marital_status, gender, bmi, snoring_rate, respiration_rate,
        body_temperature, limb_movement, blood_oxygen, eye_movement,
        sleeping_hours, heart_rate
    ]).reshape(1, -1)

    return input_data, age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate

user_input, age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate = get_user_input()

# Define stress levels and corresponding descriptions
stress_descriptions = {
    0: "No Stress",
    1: "Low Stress",
    2: "Moderate Stress",
    3: "High Stress",
    4: "Max Stress"
}

# Define a gradient of colors from lime to red
colors = ['#d0f0c0', '#b0e57c', '#f2b700', '#f77f00', '#d62839']

# Create a horizontal bar chart with five sections
fig = go.Figure()

# Add each section to the bar chart
for i in range(5):
    fig.add_trace(go.Bar(
        x=[1],
        y=[0],
        orientation='h',
        name=stress_descriptions[i],
        marker_color=colors[i],
        width=0.5,
        showlegend=False
    ))

# Update layout to arrange sections
fig.update_layout(
    barmode='stack',
    xaxis=dict(
        tickvals=[0, 1, 2, 3, 4],
        ticktext=["No Stress", "Low Stress", "Moderate Stress", "High Stress", "Max Stress"],
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(showticklabels=False, showgrid=False),
    plot_bgcolor="white",
    margin=dict(l=30, r=30, t=30, b=30),
    height=200,
    width=600
)

# Function to decode user input back to human-readable labels
def decode_user_input(age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate):
    # (This function remains unchanged)

# Predict button
if st.button("Predict Stress Level"):
    # Predict the stress level
    prediction = model.predict(user_input)[0]
    stress_level = stress_descriptions.get(prediction, "Unknown")
    
    # Add the chat bubble above the correct section
    fig.add_annotation(
        x=prediction + 0.5,  # Position the chat bubble based on the prediction
        y=0.5,
        text=f"<b>{stress_level}</b>",
        showarrow=False,
        font=dict(size=14, color="black"),
        align="center",
        bgcolor="white",
        bordercolor=colors[prediction],
        borderwidth=2,
        borderpad=4
    )
    
    # Display the updated bar chart with the chat bubble
    st.plotly_chart(fig)

    # Decode and display the interpretations of the user's input
    age_desc, bmi_desc, marital_desc, gender_desc, snoring_desc, respiration_desc, body_temp_desc, limb_desc, oxygen_desc, eye_desc, sleep_desc, heart_desc = decode_user_input(
        age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate
    )

    st.markdown(f"**Your Input Interpretation:**")
    st.write(f"Age: {age} {age_desc}", unsafe_allow_html=True)
    st.write(f"BMI: {bmi} {bmi_desc}", unsafe_allow_html=True)
    st.write(f"Marital Status: {marital_desc}")
    st.write(f"Gender: {gender_desc}")
    st.write(f"Snoring Rate: {snoring_rate} {snoring_desc}", unsafe_allow_html=True)
    st.write(f"Respiration Rate: {respiration_rate} {respiration_desc}", unsafe_allow_html=True)
    st.write(f"Body Temperature: {body_temperature} °F {body_temp_desc}", unsafe_allow_html=True)
    st.write(f"Limb Movement: {limb_movement} {limb_desc}", unsafe_allow_html=True)
    st.write(f"Blood Oxygen: {blood_oxygen} {oxygen_desc}", unsafe_allow_html=True)
    st.write(f"Eye Movement: {eye_movement} {eye_desc}", unsafe_allow_html=True)
    st.write(f"Sleeping Hours: {sleeping_hours} {sleep_desc}", unsafe_allow_html=True)
    st.markdown(f"Heart Rate: {heart_rate} {heart_desc}", unsafe_allow_html=True)

    # Display the predicted stress level below the input interpretation
    st.subheader(f"Predicted Stress Level: {stress_level} (Level {prediction})")

    # Display suggestions for stress reduction
    st.divider()
    st.caption("Meditation, massage, and a warm shower before bed can help you reduce stress when sleeping. Have a SWEET DREAM :heart::crescent_moon:")

    # Prepare data for CSV download
    data = {
        "Age": [age],
        "BMI": [bmi],
        "Marital Status": [marital_desc],
        "Gender": [gender_desc],
        "Snoring Rate": [snoring_rate],
        "Respiration Rate": [respiration_rate],
        "Body Temperature": [body_temperature],
        "Limb Movement": [limb_movement],
        "Blood Oxygen": [blood_oxygen],
        "Eye Movement": [eye_movement],
        "Sleeping Hours": [sleeping_hours],
        "Heart Rate": [heart_rate],
        "Predicted Stress Level": [stress_level]
    }
    
    df = pd.DataFrame(data)
    
    # CSV Download button
    csv = df.to_csv(index=False)
    st.download_button(label="Download Data as CSV", data=csv, file_name="stress_prediction.csv", mime="text/csv")
