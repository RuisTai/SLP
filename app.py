pip install shap

import shap
import streamlit as st

st.write(f"SHAP version: {shap.__version__}")

# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import shap  # Ensure SHAP is installed: pip install shap
# from io import BytesIO
# import base64 

# # -------------------------------
# # 1. Set Professional Background
# # -------------------------------
# def set_professional_background():
#     st.markdown(
#         """
#         <style>
#         /* Main App Background and Text Color */
#         .stApp {
#             background-color: #FAFAFA; /* Light Beige */
#             color: #212121; /* Dark Gray for text */
#         }
        
#         /* Sidebar Styling */
#         .sidebar .sidebar-content {
#             background-color: #FFFFFF; /* White Background for Sidebar */
#             padding: 20px; /* Increased padding for better spacing */
#             border-radius: 10px; /* Rounded corners for a modern look */
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
#         }
        
#         /* Sidebar Title Font */
#         .sidebar .sidebar-content h2 {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             font-size: 20px;
#             font-weight: bold;
#             margin-bottom: 15px;
#             color: #1A237E; /* Navy Blue */
#         }
        
#         /* Sidebar Text Styling */
#         .sidebar .sidebar-content p {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             font-size: 14px;
#             color: #212121; /* Dark Gray */
#         }
        
#         /* Button Styling */
#         .stButton>button {
#             background-color: #1A237E; /* Navy Blue */
#             color: white;
#             border-radius: 5px;
#             padding: 10px 20px;
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             font-size: 14px;
#             font-weight: bold;
#         }
        
#         /* Input Widget Styling */
#         /* Text Inputs */
#         .stTextInput>div>div>input {
#             background-color: #F0F0F0;
#             color: #212121;
#             border: 1px solid #B0BEC5;
#             border-radius: 5px;
#             padding: 8px;
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             font-size: 14px;
#         }
        
#         /* Number Inputs */
#         .stNumberInput>div>div>input {
#             background-color: #F0F0F0;
#             color: #212121;
#             border: 1px solid #B0BEC5;
#             border-radius: 5px;
#             padding: 8px;
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             font-size: 14px;
#         }
        
#         /* Select Boxes */
#         .stSelectbox>div>div>div>div>div>select {
#             background-color: #F0F0F0;
#             color: #212121;
#             border: 1px solid #B0BEC5;
#             border-radius: 5px;
#             padding: 8px;
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             font-size: 14px;
#         }
        
#         /* Adjusting Plotly Chart Container */
#         .plotly {
#             margin: 0 auto;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Apply the professional background and sidebar styling
# set_professional_background()

# # -------------------------------
# # 2. Load Model and Initialize SHAP
# # -------------------------------

# # Load the trained Gradient Boosting model
# model = joblib.load('gradient_boosting_model.pkl')

# # Cache the SHAP explainer using cache_resource
# @st.cache_resource
# def get_shap_explainer(_model):
#     return shap.TreeExplainer(_model)

# explainer = get_shap_explainer(model)

# # Define feature names (ensure these match the order of your model's input features)
# feature_names = [
#     'Age', 'Marital Status', 'Gender', 'BMI', 'Snoring Rate',
#     'Respiration Rate', 'Body Temperature', 'Limb Movement',
#     'Blood Oxygen', 'Eye Movement', 'Sleeping Hours', 'Heart Rate'
# ]

# # -------------------------------
# # 3. Initialize History
# # -------------------------------
# if 'history' not in st.session_state:
#     st.session_state.history = []
# if 'show_history' not in st.session_state:
#     st.session_state.show_history = False

# # -------------------------------
# # 4. Streamlit App Title
# # -------------------------------
# st.title("Stress Prediction System")

# # -------------------------------
# # 5. Sidebar for Input
# # -------------------------------
# with st.sidebar:
#     # Custom sidebar title with unique font styling
#     st.markdown("<h2>Input Features</h2>", unsafe_allow_html=True)
    
#     # Instructional text with customized font
#     st.markdown("<p>Enter the features below to predict the stress level from 0 to 4:</p>", unsafe_allow_html=True)

#     def get_user_input():
#         age = st.number_input("Age (18-80)", min_value=18, max_value=80, value=22, step=1)
#         marital_status = st.selectbox("Marital Status", options=["Yes", "No"])
#         gender = st.selectbox("Gender", options=["Male", "Female"])
#         bmi = st.number_input("BMI (18.0-40.0)", min_value=18.0, max_value=40.0, value=25.0, step=0.1)
#         snoring_rate = st.text_input("Snoring Rate (0-50)", value="")
#         respiration_rate = st.number_input("Respiration Rate (0-50)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
#         body_temperature = st.number_input("Body Temperature °F (60-110)", min_value=60.0, max_value=110.0, value=90.0, step=0.1)
#         limb_movement = st.text_input("Limb Movement (0-35)", value="")
#         blood_oxygen = st.number_input("Blood Oxygen (60-110)", min_value=60.0, max_value=110.0, value=80.0, step=0.1)
#         eye_movement = st.text_input("Eye Movement (0-35)", value="")
#         sleeping_hours = st.number_input("Sleeping Hours (0-24)", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
#         heart_rate = st.number_input("Heart Rate (30-100)", min_value=30.0, max_value=100.0, value=70.0, step=0.1)
    
#         marital_status = 1 if marital_status == "Yes" else 0
#         gender = 1 if gender == "Male" else 0
        
#         # Convert empty or invalid inputs to zero with error handling
#         try:
#             snoring_rate = float(snoring_rate) if snoring_rate else 0
#         except ValueError:
#             snoring_rate = 0
#         try:
#             limb_movement = float(limb_movement) if limb_movement else 0
#         except ValueError:
#             limb_movement = 0
#         try:
#             eye_movement = float(eye_movement) if eye_movement else 0
#         except ValueError:
#             eye_movement = 0
    
#         input_data = np.array([
#             age, marital_status, gender, bmi, snoring_rate, respiration_rate,
#             body_temperature, limb_movement, blood_oxygen, eye_movement,
#             sleeping_hours, heart_rate
#         ]).reshape(1, -1)
    
#         return input_data, age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate

#     user_input, age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate = get_user_input()

# # -------------------------------
# # 6. Define Stress Levels and Colors
# # -------------------------------
# stress_descriptions = {
#     0: "No Stress",
#     1: "Low Stress",
#     2: "Moderate Stress",
#     3: "High Stress",
#     4: "Max Stress"
# }

# colors = ['#d0f0c0', '#b0e57c', '#f2b700', '#f77f00', '#d62839']

# # -------------------------------
# # 7. Create Horizontal Bar Chart
# # -------------------------------
# fig = go.Figure()

# for i in range(5):
#     fig.add_trace(go.Bar(
#         x=[1],
#         y=[0],
#         orientation='h',
#         name=stress_descriptions[i],
#         marker_color=colors[i],
#         width=0.6,
#         showlegend=False
#     ))

# fig.update_layout(
#     barmode='stack',
#     xaxis=dict(
#         tickvals=[0, 1, 2, 3, 4],
#         ticktext=["No Stress", "Low Stress", "Moderate Stress", "High Stress", "Max Stress"],
#         showgrid=False,
#         zeroline=False
#     ),
#     yaxis=dict(showticklabels=False, showgrid=False),
#     plot_bgcolor="white",
#     margin=dict(l=20, r=20, t=20, b=20),
#     height=250,
#     width=800
# )

# # -------------------------------
# # 8. Decode User Input Function
# # -------------------------------
# def decode_user_input(age, bmi, marital_status, gender, snoring_rate, respiration_rate, body_temperature, limb_movement, blood_oxygen, eye_movement, sleeping_hours, heart_rate):
#     if age <= 18:
#         age_desc = "(Adolescent)"
#     elif age <= 24:
#         age_desc = "(Young adult)"
#     elif age <= 45:
#         age_desc = "(Adult)"
#     elif age <= 64:
#         age_desc = "(Middle age adult)"
#     else:
#         age_desc = "(Older adult)"

#     if bmi < 18.5:
#         bmi_desc = "(Underweight)"
#     elif bmi <= 24.9:
#         bmi_desc = "(Normal weight)"
#     elif bmi <= 29.9:
#         bmi_desc = "<span style='color:orange'>(Overweight)</span>"
#     elif bmi <= 30:
#         bmi_desc = "<span style='color:red'>(Obese)</span>"
#     else:
#         bmi_desc = "<span style='color:red'>(Extremely Obese)</span>"

#     marital_desc = "Married" if marital_status == 1 else "Not married"
#     gender_desc = "Male" if gender == 1 else "Female"

#     if snoring_rate <= 5:
#         snoring_desc = "(Normal)"
#     elif snoring_rate <= 15:
#         snoring_desc = "(Mild snoring)"
#     elif snoring_rate <= 30:
#         snoring_desc = "(Moderate snoring)"
#     elif snoring_rate <= 45:
#         snoring_desc = "<span style='color:orange'>(Heavy Snoring)</span>"
#     else:
#         snoring_desc = "<span style='color:red'>(Severe Snoring)</span>"

#     if respiration_rate <= 11:
#         respiration_desc = "<span style='color:orange'>(Hypoventilation-Slow Breath)</span>"
#     elif 12 <= respiration_rate <= 20:
#         respiration_desc = "(Normal)"
#     else:
#         respiration_desc = "<span style='color:red'>(Hyperventilation-Rapid Breath)</span>"

#     if body_temperature < 79:
#         body_temp_desc = "<span style='color:red'>(Hypothermia-Low)</span>"
#     elif 80 <= body_temperature <= 100:
#         body_temp_desc = "(Normal)"
#     else:
#         body_temp_desc = "<span style='color:red'>(Hyperthermia-High)</span>"

#     if limb_movement <= 5:
#         limb_desc = "(Normal)"
#     elif 6 <= limb_movement <= 25:
#         limb_desc = "(Moderate)"
#     else:
#         limb_desc = "<span style='color:red'>(Severe)</span>"

#     if blood_oxygen <= 69:
#         oxygen_desc = "<span style='color:red'>(Cyanosis-Low)</span>"
#     elif blood_oxygen <= 79:
#         oxygen_desc = "<span style='color:red'>(Severe Hypoxia)</span>"
#     elif blood_oxygen <= 89:
#         oxygen_desc = "<span style='color:orange'>(Low Oxygen Level)</span>"
#     elif 90 <= blood_oxygen <= 94:
#         oxygen_desc = "(Moderate Oxygen Level)"
#     else:
#         oxygen_desc = "(Normal Oxygen Level)"

#     if eye_movement <= 25:
#         eye_desc = "(Normal)"
#     else:
#         eye_desc = "<span style='color:red'>(High REM)</span>"

#     if sleeping_hours <= 6:
#         sleep_desc = "<span style='color:red'>(Sleep Deprivation)</span>"
#     elif 7 <= sleeping_hours <= 9:
#         sleep_desc = "(Normal)"
#     else:
#         sleep_desc = "<span style='color:red'>(Hypersomnia)</span>"

#     if heart_rate <= 39:
#         heart_desc = "<span style='color:red'>(Bradycardia-Too Slow)</span>"
#     elif 40 <= heart_rate <= 75:
#         heart_desc = "(Normal)"
#     else:
#         heart_desc = "<span style='color:red'>(Tachycardia-Too Rapid)</span>"

#     return age_desc, bmi_desc, marital_desc, gender_desc, snoring_desc, respiration_desc, body_temp_desc, limb_desc, oxygen_desc, eye_desc, sleep_desc, heart_desc

# # -------------------------------
# # 9. Predict Button Functionality
# # -------------------------------
# if st.button("Predict Stress Level"):
#     # Validate if all inputs are within the specified ranges
#     try:
#         snoring_rate_val = float(snoring_rate)
#         limb_movement_val = float(limb_movement)
#         eye_movement_val = float(eye_movement)
#     except ValueError:
#         st.error("**Error:** Please ensure that Snoring Rate, Limb Movement, and Eye Movement are numeric values.")
#     else:
#         if not (18 <= age <= 80):
#             st.error("**Error:** Please insert the Age within the range (18-80).")
#         elif not (18.0 <= bmi <= 40.0):
#             st.error("**Error:** Please insert the BMI within the range (18.0-40.0).")
#         elif not (0 <= snoring_rate_val <= 50):
#             st.error("**Error:** Please insert the Snoring Rate within the range (0-50).")
#         elif not (0 <= respiration_rate <= 50):
#             st.error("**Error:** Please insert the Respiration Rate within the range (0-50).")
#         elif not (60.0 <= body_temperature <= 110.0):
#             st.error("**Error:** Please insert the Body Temperature within the range (60.0-110.0 °F).")
#         elif not (0 <= limb_movement_val <= 35):
#             st.error("**Error:** Please insert the Limb Movement within the range (0-35).")
#         elif not (60 <= blood_oxygen <= 110):
#             st.error("**Error:** Please insert the Blood Oxygen within the range (60-110).")
#         elif not (0 <= eye_movement_val <= 35):
#             st.error("**Error:** Please insert the Eye Movement within the range (0-35).")
#         elif not (0 <= sleeping_hours <= 24):
#             st.error("**Error:** Please insert the Sleeping Hours within the range (0-24).")
#         elif not (30 <= heart_rate <= 100):
#             st.error("**Error:** Please insert the Heart Rate within the range (30-100).")
#         else:
#             # If inputs are valid, predict the stress level
#             prediction = model.predict(user_input)[0]
#             stress_level = stress_descriptions.get(prediction, "Unknown")
        
#             # Check if any of the optional inputs are empty or zero
#             incomplete_data_warning = ""
#             if snoring_rate_val == 0 or limb_movement_val == 0 or eye_movement_val == 0:
#                 incomplete_data_warning = (
#                     "<span style='color:#d61e40'>Note: Some of the input variables (Snoring Rate, Limb Movement, Eye Movement) were not provided or are zero.</span>"
#                     "<span style='color:#d61e40'> The prediction may not be highly accurate due to incomplete data.</span>"
#                 )
        
#             # Add the chat bubble above the correct section
#             fig.add_annotation(
#                 x=prediction + 0.5,  # Position the chat bubble based on the prediction
#                 y=0.5,
#                 text=f"<b>{stress_level}</b>",
#                 showarrow=True,
#                 arrowhead=2,
#                 ax=0,
#                 ay=-40,
#                 font=dict(size=15, color="black"),
#                 align="center",
#                 bgcolor="white",
#                 bordercolor=colors[prediction],
#                 borderwidth=2,
#                 borderpad=4
#             )
        
#             # Display the updated bar chart with the chat bubble
#             st.plotly_chart(fig)
        
#             # Decode and display the interpretations of the user's input
#             age_desc, bmi_desc, marital_desc, gender_desc, snoring_desc, respiration_desc, body_temp_desc, limb_desc, oxygen_desc, eye_desc, sleep_desc, heart_desc = decode_user_input(
#                 age, bmi, marital_status, gender, snoring_rate_val, respiration_rate, body_temperature, limb_movement_val, blood_oxygen, eye_movement_val, sleeping_hours, heart_rate
#             )

#             # Display interpretations with HTML styling
#             st.markdown("## **Your Input Interpretation:**")
#             st.markdown(f"**Age:** {age} {age_desc}", unsafe_allow_html=True)
#             st.markdown(f"**BMI:** {bmi} {bmi_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Marital Status:** {marital_desc}")
#             st.markdown(f"**Gender:** {gender_desc}")
#             st.markdown(f"**Snoring Rate:** {snoring_rate_val} {snoring_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Respiration Rate:** {respiration_rate} {respiration_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Body Temperature:** {body_temperature} °F {body_temp_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Limb Movement:** {limb_movement_val} {limb_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Blood Oxygen:** {blood_oxygen} {oxygen_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Eye Movement:** {eye_movement_val} {eye_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Sleeping Hours:** {sleeping_hours} {sleep_desc}", unsafe_allow_html=True)
#             st.markdown(f"**Heart Rate:** {heart_rate} {heart_desc}", unsafe_allow_html=True)
        
#             if incomplete_data_warning:
#                 st.markdown(f"**Warning:** {incomplete_data_warning}", unsafe_allow_html=True)
        
#             # Compute SHAP values for the user input
#             shap_values = explainer.shap_values(user_input)
        
#             # SHAP Bar Plot for feature importance
#             shap_df = pd.DataFrame({
#                 'Feature': feature_names,
#                 'SHAP Value': shap_values[prediction][0]  # Adjust indexing as needed
#             })

#             # Sort features by absolute SHAP value
#             shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
#             shap_df = shap_df.sort_values(by='Abs SHAP', ascending=True)

#             # Create a horizontal bar chart for SHAP values
#             fig_shap = go.Figure()

#             fig_shap.add_trace(go.Bar(
#                 x=shap_df['SHAP Value'],
#                 y=shap_df['Feature'],
#                 orientation='h',
#                 marker_color='rgba(58, 71, 80, 0.6)',
#                 hoverinfo='x+y',
#             ))

#             fig_shap.update_layout(
#                 title='Feature Importance for Prediction',
#                 xaxis_title='SHAP Value',
#                 yaxis_title='Feature',
#                 height=400,
#                 margin=dict(l=150, r=50, t=50, b=50),
#                 plot_bgcolor='white',
#                 showlegend=False
#             )

#             # Display the SHAP bar plot in Streamlit
#             st.plotly_chart(fig_shap, use_container_width=True)

#             # Save user input and prediction to history
#             st.session_state.history.append({
#                 "Age": age,
#                 "BMI": bmi,
#                 "Marital Status": marital_desc,
#                 "Gender": gender_desc,
#                 "Snoring Rate": snoring_rate_val,
#                 "Respiration Rate": respiration_rate,
#                 "Body Temperature": body_temperature,
#                 "Limb Movement": limb_movement_val,
#                 "Blood Oxygen": blood_oxygen,
#                 "Eye Movement": eye_movement_val,
#                 "Sleeping Hours": sleeping_hours,
#                 "Heart Rate": heart_rate,
#                 "Stress Level": stress_level
#             })

# # -------------------------------
# # 10. Prediction History Section
# # -------------------------------
# st.subheader("Prediction History")

# # Button to toggle showing prediction history
# if st.button("Show Prediction History"):
#     st.session_state.show_history = not st.session_state.show_history  # Toggle history visibility

# # Display prediction history if the button was clicked and history exists
# if st.session_state.show_history and st.session_state.history:
#     df_history = pd.DataFrame(st.session_state.history)
#     st.write(df_history)
    
#     # Create a downloadable CSV file
#     def create_download_link(df, filename="history.csv"):
#         csv = df.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'

#     st.markdown(create_download_link(df_history), unsafe_allow_html=True)
