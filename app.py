import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration safely
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Render Header Elements safely
st.title("🎓 Student Exam Score Predictor")
st.caption("Predict final exam scores using historical performance and behavioral traits.")
st.write("---")

# Model Loading Pipeline
MODEL_PATH = "model(2).pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Error: `{MODEL_PATH}` not found in the root repository. Please check your file upload structure.")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Failed to parse serialization object: {e}")
        return None

model = load_model()

if model is not None:
    st.subheader("📊 Input Student Metrics")
    
    # Grid Layout Layout Partitioning
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = col1.slider(
            "📚 Study Hours (Per Day)", 
            min_value=1.0, max_value=12.0, value=6.0, step=0.5,
            help="Average duration of study invested in preparation for the exam."
        )
        attendance_percent = col1.slider(
            "🏫 Attendance Rate (%)", 
            min_value=50.0, max_value=100.0, value=85.0, step=0.1,
            help="Classroom attendance percentage."
        )

    with col2:
        sleep_hours = col2.slider(
            "😴 Sleep Hours (Per Night)", 
            min_value=4.0, max_value=9.0, value=7.0, step=0.5,
            help="Average daily sleep duration."
        )
        previous_scores = col2.slider(
            "📈 Previous Academic Score", 
            min_value=40, max_value=95, value=70, step=1,
            help="Average historical scores achieved in past examinations."
        )

    st.write("---")
    
    # Active Inference Trigger
    predict_btn = st.button("🚀 Predict Final Exam Score", use_container_width=True, type="primary")

    if predict_btn:
        # Map parameters to features array
        features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
        
        try:
            prediction = model.predict(features)[0]
            st.success("✨ Prediction Calculated Successfully!")
            
            # Metric Card Representation
            st.metric(label="Estimated Final Exam Score", value=f"{prediction:.2f} / 100")
            
        except Exception as e:
            st.error(f"Inference pipeline execution error encountered: {e}")
