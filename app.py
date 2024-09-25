import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import base64 
import os

# -------------------------------
# 1. Set Professional Background
# -------------------------------
def set_professional_background():
    st.markdown(
        """
        <style>
        /* Main App Background and Text Color */
        .stApp {
            background-color: #FAFAFA; /* Light Beige */
            color: #212121; /* Dark Gray for text */
        }
        
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #FFFFFF; /* White Background for Sidebar */
            padding: 20px; /* Increased padding for better spacing */
            border-radius: 10px; /* Rounded corners for a modern look */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        
        /* Sidebar Title Font */
        .sidebar .sidebar-content h2 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #1A237E; /* Navy Blue */
        }
        
        /* Sidebar Text Styling */
        .sidebar .sidebar-content p {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            color: #212121; /* Dark Gray */
        }
        
        /* Button Styling */
        .stButton>button {
            background-color: #1A237E; /* Navy Blue */
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            font-weight: bold;
        }
        
        /* Input Widget Styling */
        /* Text Inputs */
        .stTextInput>div>div>input {
            background-color: #F0F0F0;
            color: #212121;
            border: 1px solid #B0BEC5;
            border-radius: 5px;
            padding: 8px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
        }
        
        /* Number Inputs */
        .stNumberInput>div>div>input {
            background-color: #F0F0F0;
            color: #212121;
            border: 1px solid #B0BEC5;
            border-radius: 5px;
            padding: 8px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
        }
        
        /* Select Boxes */
        .stSelectbox>div>div>div>div>div>select {
            background-color: #F0F0F0;
            color: #212121;
            border: 1px solid #B0BEC5;
            border-radius: 5px;
            padding: 8px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
        }
        
        /* Adjusting Plotly Chart Container */
        .plotly {
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the professional background and sidebar styling
set_professional_background()

# -------------------------------
# 2. Load Model with Caching
# -------------------------------

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"**Error:** Model file '{model_path}' not found. Please ensure it exists in the specified path.")
        st.stop()
    return joblib.load(model_path)

# Specify the path to your model
model_path = 'gradient_boosting_model.pkl'

# Load the trained Gradient Boosting model
model = load_model(model_path)

# Define feature names (ensure these match the order of your model's input features)
feature_names = [
    'Age', 'Marital Status', 'Gender', 'BMI', 'Snoring Rate',
    'Respiration Rate', 'Body Temperature', 'Limb Movement',
    'Blood Oxygen', 'Eye Movement', 'Sleeping Hours', 'Heart Rate'
]

# -------------------------------
# 3. Initialize History
# -------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# -------------------------------
# 4. Streamlit App Title
# -------------------------------
st.title("🧠 **Stress Prediction System**")

# -------------------------------
# 5. Sidebar for Input with Help Texts
# -------------------------------
with st.sidebar:
    # Custom sidebar title with unique font styling
    st.markdown("<h2>📋 Input Features</h2>", unsafe_allow_html=True)
    
    # Instructional text with customized font
    st.markdown("<p>Enter the features below to predict your stress level from 0 to 4:</p>", unsafe_allow_html=True)

    def get_user_input():
        age = st.number_input(
            "Age (18-80)",
            min_value=18,
            max_value=80,
            value=22,
            step=1,
            help="Enter your age in years (must be between 18 and 80)."
        )
        marital_status = st.selectbox(
            "Marital Status",
            options=["Yes", "No"],
            help="Select 'Yes' if you are married, otherwise 'No'."
        )
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Select your gender."
        )
        bmi = st.number_input(
            "BMI (18.0-40.0)",
            min_value=18.0,
            max_value=40.0,
            value=25.0,
            step=0.1,
            help="Enter your Body Mass Index (BMI) value."
        )
        snoring_rate = st.text_input(
            "Snoring Rate (0-50)",
            value="",
            help="Enter your snoring rate. If unsure, leave blank or enter 0."
        )
        respiration_rate = st.number_input(
            "Respiration Rate (0-50)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.1,
            help="Enter your respiration rate."
        )
        body_temperature = st.number_input(
            "Body Temperature °F (60-110)",
            min_value=60.0,
            max_value=110.0,
            value=90.0,
            step=0.1,
            help="Enter your body temperature in Fahrenheit."
        )
        limb_movement = st.text_input(
            "Limb Movement (0-35)",
            value="",
            help="Enter your limb movement rate. If unsure, leave blank or enter 0."
        )
        blood_oxygen = st.number_input(
            "Blood Oxygen (60-110)",
            min_value=60.0,
            max_value=110.0,
            value=80.0,
            step=0.1,
            help="Enter your blood oxygen level."
        )
        eye_movement = st.text_input(
            "Eye Movement (0-35)",
            value="",
            help="Enter your eye movement rate. If unsure, leave blank or enter 0."
        )
        sleeping_hours = st.number_input(
            "Sleeping Hours (0-24)",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.1,
            help="Enter the number of hours you slept."
        )
        heart_rate = st.number_input(
            "Heart Rate (30-100)",
            min_value=30.0,
            max_value=100.0,
            value=70.0,
            step=0.1,
            help="Enter your heart rate."
        )
    
        marital_status = 1 if marital_status == "Yes" else 0
        gender = 1 if gender == "Male" else 0
        
        # Convert empty or invalid inputs to zero with error handling
        try:
            snoring_rate = float(snoring_rate) if snoring_rate else 0
        except ValueError:
            snoring_rate = 0
        try:
            limb_movement = float(limb_movement) if limb_movement else 0
        except ValueError:
            limb_movement = 0
        try:
            eye_movement = float(eye_movement) if eye_movement else 0
        except ValueError:
            eye_movement = 0
    
        input_data = np.array([
            age, marital_status, gender, bmi, snoring_rate, respiration_rate,
            body_temperature, limb_movement, blood_oxygen, eye_movement,
            sleeping_hours, heart_rate
        ]).reshape(1, -1)
    
        return input_data, age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate

    user_input, age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate = get_user_input()

# -------------------------------
# 6. Define Stress Levels and Colors
# -------------------------------
stress_descriptions = {
    0: "No Stress",
    1: "Low Stress",
    2: "Moderate Stress",
    3: "High Stress",
    4: "Max Stress"
}

colors = ['#d0f0c0', '#b0e57c', '#f2b700', '#f77f00', '#d62839']

# -------------------------------
# 7. Create Horizontal Bar Chart
# -------------------------------
fig = go.Figure()

for i in range(5):
    fig.add_trace(go.Bar(
        x=[1],
        y=[0],
        orientation='h',
        name=stress_descriptions[i],
        marker_color=colors[i],
        width=0.6,
        showlegend=False
    ))

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
    margin=dict(l=20, r=20, t=20, b=20),
    height=250,
    width=800
)

# -------------------------------
# 8. Decode User Input Function
# -------------------------------
def decode_user_input(age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate):
    if age <= 18:
        age_desc = "(Adolescent)"
    elif age <= 24:
        age_desc = "(Young adult)"
    elif age <= 45:
        age_desc = "(Adult)"
    elif age <= 64:
        age_desc = "(Middle age adult)"
    else:
        age_desc = "(Older adult)"

    if bmi < 18.5:
        bmi_desc = "(Underweight)"
    elif bmi <= 24.9:
        bmi_desc = "(Normal weight)"
    elif bmi <= 29.9:
        bmi_desc = "(Overweight)"
    elif bmi <= 30:
        bmi_desc = "(Obese)"
    else:
        bmi_desc = "(Extremely Obese)"

    marital_desc = "Married" if marital_status == 1 else "Not married"
    gender_desc = "Male" if gender == 1 else "Female"

    if snoring_rate <= 5:
        snoring_desc = "(Normal)"
    elif snoring_rate <= 15:
        snoring_desc = "(Mild snoring)"
    elif snoring_rate <= 30:
        snoring_desc = "(Moderate snoring)"
    elif snoring_rate <= 45:
        snoring_desc = "(Heavy Snoring)"
    else:
        snoring_desc = "(Severe Snoring)"

    if respiration_rate <= 11:
        respiration_desc = "(Hypoventilation-Slow Breath)"
    elif 12 <= respiration_rate <= 20:
        respiration_desc = "(Normal)"
    else:
        respiration_desc = "(Hyperventilation-Rapid Breath)"

    if body_temperature < 97.0:
        body_temp_desc = "(Hypothermia-Low)"
    elif 97.0 <= body_temperature <= 99.5:
        body_temp_desc = "(Normal)"
    else:
        body_temp_desc = "(Hyperthermia-High)"

    if limb_movement <= 5:
        limb_desc = "(Normal)"
    elif 6 <= limb_movement <= 25:
        limb_desc = "(Moderate)"
    else:
        limb_desc = "(Severe)"

    if blood_oxygen <= 69:
        oxygen_desc = "(Cyanosis-Low)"
    elif blood_oxygen <= 79:
        oxygen_desc = "(Severe Hypoxia)"
    elif blood_oxygen <= 89:
        oxygen_desc = "(Low Oxygen Level)"
    elif 90 <= blood_oxygen <= 94:
        oxygen_desc = "(Moderate Oxygen Level)"
    else:
        oxygen_desc = "(Normal Oxygen Level)"

    if eye_movement <= 25:
        eye_desc = "(Normal)"
    else:
        eye_desc = "(High REM)"

    if sleeping_hours <= 6:
        sleep_desc = "(Sleep Deprivation)"
    elif 7 <= sleeping_hours <= 9:
        sleep_desc = "(Normal)"
    else:
        sleep_desc = "(Hypersomnia)"

    if heart_rate <= 39:
        heart_desc = "(Bradycardia-Too Slow)"
    elif 40 <= heart_rate <= 75:
        heart_desc = "(Normal)"
    else:
        heart_desc = "(Tachycardia-Too Rapid)"

    return age_desc, bmi_desc, marital_desc, gender_desc, snoring_desc, respiration_desc, body_temp_desc, limb_desc, oxygen_desc, eye_desc, sleep_desc, heart_desc

# -------------------------------
# 9. Recommendations Function
# -------------------------------
def provide_recommendations(
    bmi, blood_oxygen, heart_rate, 
    snoring_rate, respiration_rate, body_temperature, 
    limb_movement, eye_movement, sleeping_hours
):
    recommendations = []
    
    # BMI Recommendations
    if bmi < 18.5:
        recommendations.append("📉 **Underweight**: Consider consulting a healthcare provider for a nutritional plan to reach a healthier weight.")
    elif bmi <= 24.9:
        recommendations.append("✅ **Normal Weight**: Great job maintaining a healthy BMI!")
    elif bmi <= 29.9:
        recommendations.append("⚖️ **Overweight**: Engaging in a balanced diet and regular physical activity can help manage your BMI.")
    else:
        recommendations.append("📈 **Obese**: It's advisable to seek guidance from a healthcare professional for a comprehensive weight management plan.")
    
    # Blood Oxygen Recommendations
    if blood_oxygen < 90:
        recommendations.append("🩸 **Low Blood Oxygen**: Low blood oxygen levels detected. Please consult a healthcare professional immediately.")
    elif blood_oxygen <= 94:
        recommendations.append("🟠 **Low Oxygen Level**: Consider deep breathing exercises and ensure you're in a well-ventilated environment.")
    else:
        recommendations.append("🟢 **Normal Blood Oxygen**: Your blood oxygen levels are within the normal range.")
    
    # Heart Rate Recommendations
    if heart_rate < 40:
        recommendations.append("❤️ **Bradycardia**: Abnormally low heart rate detected. Consider seeking medical advice.")
    elif heart_rate <= 75:
        recommendations.append("🟢 **Normal Heart Rate**: Your heart rate is within the normal range.")
    else:
        recommendations.append("❤️ **Tachycardia**: Abnormally high heart rate detected. It might be beneficial to engage in relaxation techniques or consult a healthcare provider.")
    
    # Snoring Rate Recommendations
    if snoring_rate > 30:
        recommendations.append("😴 **Heavy Snoring**: Persistent heavy snoring may indicate sleep apnea. Consider consulting a sleep specialist.")
    elif snoring_rate > 15:
        recommendations.append("😴 **Mild Snoring**: Moderate snoring can disrupt your sleep. Maintaining a healthy weight and avoiding alcohol before bedtime might help.")
    else:
        recommendations.append("✅ **Normal Snoring**: Your snoring rate is within the normal range.")
    
    # Respiration Rate Recommendations
    if respiration_rate < 12:
        recommendations.append("🌬️ **Slow Respiration**: A lower respiration rate may indicate hypoventilation. Consider breathing exercises.")
    elif respiration_rate > 20:
        recommendations.append("🌬️ **Rapid Respiration**: A higher respiration rate may indicate hyperventilation. Practice relaxation techniques.")
    else:
        recommendations.append("🟢 **Normal Respiration Rate**: Your respiration rate is within the normal range.")
    
    # Body Temperature Recommendations
    if body_temperature < 97.0:
        recommendations.append("🌡️ **Low Body Temperature**: Consider dressing warmly and consulting a healthcare provider if you feel unwell.")
    elif body_temperature > 99.5:
        recommendations.append("🌡️ **High Body Temperature**: Stay hydrated and consider seeking medical attention if the temperature persists.")
    else:
        recommendations.append("🟢 **Normal Body Temperature**: Your body temperature is within the normal range.")
    
    # Limb Movement Recommendations
    if limb_movement > 25:
        recommendations.append("🦵 **Severe Limb Movement**: Excessive limb movement during sleep may affect sleep quality. Consider relaxation techniques before bedtime.")
    elif limb_movement > 5:
        recommendations.append("🦵 **Moderate Limb Movement**: Some limb movement is normal, but excessive movement can disrupt sleep.")
    else:
        recommendations.append("✅ **Normal Limb Movement**: Your limb movement during sleep is within the normal range.")
    
    # Eye Movement Recommendations
    if eye_movement > 25:
        recommendations.append("👁️ **High REM Activity**: Elevated eye movement during sleep can be associated with stress. Consider stress-reduction techniques.")
    else:
        recommendations.append("🟢 **Normal Eye Movement**: Your eye movement during sleep is within the normal range.")
    
    # Sleeping Hours Recommendations
    if sleeping_hours < 6:
        recommendations.append("🛌 **Sleep Deprivation**: Aim for 7-9 hours of sleep for optimal health. Consider establishing a regular sleep schedule.")
    elif sleeping_hours > 9:
        recommendations.append("🛌 **Excessive Sleep**: Consistently sleeping more than 9 hours may affect your daily routine. Aim for 7-9 hours of sleep.")
    else:
        recommendations.append("🟢 **Normal Sleeping Hours**: Your sleep duration is within the recommended range.")
    
    return recommendations

# -------------------------------
# 10. Predict Button Functionality
# -------------------------------
if st.button("Predict Stress Level"):
    # Validate if all inputs are within the specified ranges
    try:
        snoring_rate_val = float(snoring_rate)
        limb_movement_val = float(limb_movement)
        eye_movement_val = float(eye_movement)
    except ValueError:
        st.error("**Error:** Please ensure that Snoring Rate, Limb Movement, and Eye Movement are numeric values.")
    else:
        # Validate input ranges
        if not (18 <= age <= 80):
            st.error("**Error:** Please insert the Age within the range (18-80).")
        elif not (18.0 <= bmi <= 40.0):
            st.error("**Error:** Please insert the BMI within the range (18.0-40.0).")
        elif not (0 <= snoring_rate_val <= 50):
            st.error("**Error:** Please insert the Snoring Rate within the range (0-50).")
        elif not (0 <= respiration_rate <= 50):
            st.error("**Error:** Please insert the Respiration Rate within the range (0-50).")
        elif not (60.0 <= body_temperature <= 110.0):
            st.error("**Error:** Please insert the Body Temperature within the range (60.0-110.0 °F).")
        elif not (0 <= limb_movement_val <= 35):
            st.error("**Error:** Please insert the Limb Movement within the range (0-35).")
        elif not (60 <= blood_oxygen <= 110):
            st.error("**Error:** Please insert the Blood Oxygen within the range (60-110).")
        elif not (0 <= eye_movement_val <= 35):
            st.error("**Error:** Please insert the Eye Movement within the range (0-35).")
        elif not (0 <= sleeping_hours <= 24):
            st.error("**Error:** Please insert the Sleeping Hours within the range (0-24).")
        elif not (30 <= heart_rate <= 100):
            st.error("**Error:** Please insert the Heart Rate within the range (30-100).")
        else:
            # If inputs are valid, predict the stress level
            prediction = model.predict(user_input)[0]
            stress_level = stress_descriptions.get(prediction, "Unknown")
        
            # Check if any of the optional inputs are empty or zero
            incomplete_data_warning = ""
            if snoring_rate_val == 0 or limb_movement_val == 0 or eye_movement_val == 0:
                incomplete_data_warning = (
                    "<span style='color:#d32f2f'>⚠️ 
                    Note: Some of the input variables (Snoring Rate, Limb Movement, Eye Movement) were not provided or are zero.</span><br>"
                    "<span style='color:#d32f2f'>The prediction may not be highly accurate due to incomplete data.</span>"
                )
        
            # Add the chat bubble above the correct section
            fig.add_annotation(
                x=prediction + 0.5,  # Position the chat bubble based on the prediction
                y=0.5,
                text=f"<b>{stress_level}</b>",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                font=dict(size=15, color="black"),
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
                age, bmi, marital_status, gender, snoring_rate_val, respiration_rate, body_temperature, limb_movement_val, blood_oxygen, eye_movement_val, sleeping_hours, heart_rate
            )

            def get_status_color(description):
                critical_keywords = ["Underweight", "Hypoventilation", "Bradycardia", "Hypothermia", "Sleep Deprivation"]
                warning_keywords = ["Overweight", "Low Oxygen Level", "Severe Hypoxia", "Hyperventilation", "Hypersomnia"]
                normal_keywords = ["Normal"]

                if any(keyword in description for keyword in critical_keywords):
                    return "#d32f2f"  # Red
                elif any(keyword in description for keyword in warning_keywords):
                    return "#f57c00"  # Orange
                elif any(keyword in description for keyword in normal_keywords):
                    return "#388e3c"  # Green
                else:
                    return "#616161"  # Grey for unknown or neutral

            # Display interpretations with structured layout and accessible colors
            st.markdown("## **📝 Your Input Interpretation:**")
            st.markdown(f"** **")
            
            # # Create three columns
            col1, col2 = st.columns(2)

            with col1:
                age_color = get_status_color(age_desc)
                st.markdown(
                    f"**🧑 Age:** {age} <span style='color:{age_color}'>{age_desc}</span>",
                    unsafe_allow_html=True
                )
                bmi_color = get_status_color(bmi_desc)
                st.markdown(
                    f"**⚖️ BMI:** {bmi} <span style='color:{bmi_color}'>{bmi_desc}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**💍 Marital Status:** {marital_desc}")
                st.markdown(f"**♂️ Gender:** {gender_desc}")
                snoring_color = get_status_color(snoring_desc)
                st.markdown(
                    f"**😴 Snoring Rate:** {snoring_rate_val} <span style='color:{snoring_color}'>{snoring_desc}</span>",
                    unsafe_allow_html=True
                )
                respiration_color = get_status_color(respiration_desc)
                st.markdown(
                    f"**🌬️ Respiration Rate:** {respiration_rate} <span style='color:{respiration_color}'>{respiration_desc}</span>",
                    unsafe_allow_html=True
                )
                body_temp_color = get_status_color(body_temp_desc)
                st.markdown(
                    f"**🌡️ Body Temperature:** {body_temperature} °F <span style='color:{body_temp_color}'>{body_temp_desc}</span>",
                    unsafe_allow_html=True
                )
                limb_color = get_status_color(limb_desc)
                st.markdown(
                    f"**🦵 Limb Movement:** {limb_movement_val} <span style='color:{limb_color}'>{limb_desc}</span>",
                    unsafe_allow_html=True
                )
                oxygen_color = get_status_color(oxygen_desc)
                st.markdown(
                    f"**🩸 Blood Oxygen:** {blood_oxygen} <span style='color:{oxygen_color}'>{oxygen_desc}</span>",
                    unsafe_allow_html=True
                )
                eye_color = get_status_color(eye_desc)
                st.markdown(
                    f"**👁️ Eye Movement:** {eye_movement_val} <span style='color:{eye_color}'>{eye_desc}</span>",
                    unsafe_allow_html=True
                )
                sleep_color = get_status_color(sleep_desc)
                st.markdown(
                    f"**🛌 Sleeping Hours:** {sleeping_hours} <span style='color:{sleep_color}'>{sleep_desc}</span>",
                    unsafe_allow_html=True
                )
                heart_color = get_status_color(heart_desc)
                st.markdown(
                    f"**❤️ Heart Rate:** {heart_rate} <span style='color:{heart_color}'>{heart_desc}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"** **")

                # Display warning if incomplete data
            if incomplete_data_warning:
                st.markdown(f"**⚠️ Warning:** {incomplete_data_warning}", unsafe_allow_html=True)
                st.markdown(f"** **")

            with col2:
                # st.markdown(f"****")
                
            # Expander for additional details with accessible colors
            with st.expander("📊 View Recommendations"):
                recommendations = provide_recommendations(
                bmi, blood_oxygen, heart_rate, 
                snoring_rate_val, respiration_rate, body_temperature, 
                limb_movement_val, eye_movement_val, sleeping_hours
            )
                
                st.markdown("## **💡 Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
        
            # Save user input and prediction to history
            st.session_state.history.append({
                "Age": age,
                "BMI": bmi,
                "Marital Status": marital_desc,
                "Gender": gender_desc,
                "Snoring Rate": snoring_rate_val,
                "Respiration Rate": respiration_rate,
                "Body Temperature": body_temperature,
                "Limb Movement": limb_movement_val,
                "Blood Oxygen": blood_oxygen,
                "Eye Movement": eye_movement_val,
                "Sleeping Hours": sleeping_hours,
                "Heart Rate": heart_rate,
                "Stress Level": stress_level
            })

# -------------------------------
# 11. Prediction History Section
# -------------------------------
st.markdown("## 🕒 **Prediction History**")

# Button to toggle showing prediction history with unique key
if st.button("Show Prediction History", key="show_history_btn"):
    st.session_state.show_history = not st.session_state.show_history  # Toggle history visibility

# Display prediction history if the button was clicked and history exists
if st.session_state.show_history:
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.write(df_history)
        
        # Create a downloadable CSV file
        def create_download_link(df, filename="history.csv"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV file</a>'

        st.markdown(create_download_link(df_history), unsafe_allow_html=True)
    else:
        st.info("No predictions made yet.")
