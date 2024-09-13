import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

# Initialize an empty list to store input history
if 'input_history' not in st.session_state:
    st.session_state['input_history'] = []

# Streamlit app title
st.title("Stress Prediction System")

# Sidebar for input section with dropdown
with st.sidebar:

    # Dropdown to select between functions
    selected_function = st.selectbox("Choose an action:", ["Predict Stress Level", "Download Input History"])

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

    if selected_function == "Predict Stress Level":
        # Main function to predict stress level
        user_input, age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate = get_user_input()

        # Store user input in session state
        st.session_state['input_history'].append({
            "Age": age, "BMI": bmi, "Marital Status": marital_status, "Gender": gender,
            "Snoring Rate": snoring_rate, "Respiration Rate": respiration_rate,
            "Body Temperature": body_temperature, "Limb Movement": limb_movement,
            "Blood Oxygen": blood_oxygen, "Eye Movement": eye_movement,
            "Sleeping Hours": sleeping_hours, "Heart Rate": heart_rate
        })

        # Button to predict the stress level
        if st.button("Predict Stress Level"):
            # Prediction and results moved outside sidebar
            prediction = model.predict(user_input)[0]
            stress_descriptions = {
                0: "No Stress",
                1: "Low Stress",
                2: "Moderate Stress",
                3: "High Stress",
                4: "Max Stress"
            }

            # Define colors and create chart
            colors = ['#d0f0c0', '#b0e57c', '#f2b700', '#f77f00', '#d62839']
            fig = go.Figure()
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

            fig.add_annotation(
                x=prediction + 0.5,
                y=0.5,
                text=f"<b>{stress_descriptions[prediction]}</b>",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="center",
                bgcolor="white",
                bordercolor=colors[prediction],
                borderwidth=2,
                borderpad=4
            )

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
# Display the chart and result on the main page (not sidebar
        st.plotly_chart(fig)
        st.subheader(f"Predicted Stress Level: {stress_descriptions[prediction]} (Level {prediction})")

    elif selected_function == "Download Input History":
        # Convert input history to DataFrame
        input_history_df = pd.DataFrame(st.session_state['input_history'])

        if not input_history_df.empty:
            # Provide option to download input history as CSV
            csv = input_history_df.to_csv(index=False)
            st.download_button(
                label="Download Input History as CSV",
                data=csv,
                file_name='input_history.csv',
                mime='text/csv'
            )
        else:
            st.write("No input history recorded yet.")
