import streamlit as plt
import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom header styling
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🎓 Student Exam Score Predictor</h1>", unsafe_html=True)
st.markdown("<p style='text-align: center; color: #6B7280;'>Predict final exam scores using historical performance and behavioral traits.</p>", unsafe_html=True)
st.write("---")

# Load the trained pickle model safely
MODEL_PATH = "model(2).pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Error: `{MODEL_PATH}` not found in the current directory! Please upload it.")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

if model is not None:
    st.subheader("📊 Input Student Data")
    
    # Layout structured inputs with columns
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.slider(
            "📚 Study Hours (Per Day)", 
            min_value=1.0, max_value=12.0, value=6.0, step=0.5,
            help="Average duration of study invested in preparation for the exam."
        )
        attendance_percent = st.slider(
            "🏫 Attendance Rate (%)", 
            min_value=50.0, max_value=100.0, value=85.0, step=0.1,
            help="Classroom attendance percentage."
        )

    with col2:
        sleep_hours = st.slider(
            "😴 Sleep Hours (Per Night)", 
            min_value=4.0, max_value=9.0, value=7.0, step=0.5,
            help="Average daily sleep duration."
        )
        previous_scores = st.slider(
            "📈 Previous Academic Score", 
            min_value=40, max_value=95, value=70, step=1,
            help="Average historical scores achieved in past examinations."
        )

    st.write("---")
    
    # Center-aligned Prediction Button
    left_co, cent_co, last_co = st.columns([1, 2, 1])
    with cent_co:
        predict_btn = st.button("🚀 Predict Final Exam Score", use_container_width=True, type="primary")

    if predict_btn:
        # Arrange inputs into the exact format and ordering required by scikit-learn
        features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
        
        try:
            # Perform prediction
            prediction = model.predict(features)[0]
            
            # Display Result Layout
            st.success("✨ Prediction Calculated Successfully!")
            
            # Format nicely as a KPI metric card
            st.markdown(
                f"""
                <div style="background-color:#F3F4F6; padding:20px; border-radius:10px; border-left: 5px solid #1E3A8A; text-align:center;">
                    <p style="margin:0; font-size:18px; color:#4B5563; font-weight:bold;">Estimated Exam Score</p>
                    <h1 style="margin:0; font-size:48px; color:#1E3A8A;">{prediction:.2f} / 100</h1>
                </div>
                """, 
                unsafe_html=True
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
