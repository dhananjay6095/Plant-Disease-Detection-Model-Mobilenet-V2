import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Plant Disease Recognition", layout="centered")

# -----------------------------
# Load model safely
# -----------------------------
MODEL_PATH = "plant_disease_model.keras"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully ✅")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.warning(f"Model file not found at '{MODEL_PATH}'! Please upload the correct model.")

# -----------------------------
# Prediction function
# -----------------------------
def model_prediction(image_file):
    if model is None:
        st.error("Model is not loaded!")
        return None

    try:
        image = Image.open(image_file).convert("RGB")
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["Home", "About", "Disease Recognition"])


if app_mode == "Home":
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 38px;
            color: #2E8B57;
            font-weight: bold;
            text-align: center;
        }
        .sub-header {
            font-size: 20px;
            text-align: center;
            color: #555555;
        }
        .steps {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
        }
        .step {
            font-size: 18px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="main-header">🌿 Plant Disease Recognition System 🌿</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered tool to detect plant diseases instantly!</div>', unsafe_allow_html=True)

    # Add a two-column layout
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image("https://cdn.mos.cms.futurecdn.net/CKiu992oijYxcPJVyuuCWL-1280-80.jpg.webp", use_column_width=True)
    with col2:
        st.markdown('<div class="steps">', unsafe_allow_html=True)
        st.markdown('<div class="step">1️⃣ Upload a leaf image of your plant</div>', unsafe_allow_html=True)
        st.markdown('<div class="step">2️⃣ The AI model analyzes the leaf</div>', unsafe_allow_html=True)
        st.markdown('<div class="step">3️⃣ Get the predicted disease instantly</div>', unsafe_allow_html=True)
        st.markdown('<div class="step">4️⃣ Take necessary actions to protect your crop!</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("💡 **Tip:** Navigate to **Disease Recognition** in the sidebar to try it out!")
