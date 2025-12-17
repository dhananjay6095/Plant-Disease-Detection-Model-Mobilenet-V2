
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
        image = image.resize((224,224))
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

# -----------------------------
# Home Page
# -----------------------------
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM 🌿")
    image_path="https://cdn.mos.cms.futurecdn.net/CKiu992oijYxcPJVyuuCWL-1280-80.jpg.webp"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System!

    **How It Works:**
    1️⃣ Upload a plant leaf image
    2️⃣ The model analyzes it
    3️⃣ Get the predicted disease class

    Navigate to **Disease Recognition** to try it out!
    """)

# -----------------------------
# About Page
# -----------------------------
elif app_mode == "About":
    st.header("About Dataset & Project")
    st.markdown("""
    This dataset contains 87K images of healthy & diseased crop leaves
    categorized into 38 classes. It is split into 80/20 training & validation.

    Our project aims to help farmers and researchers quickly detect plant diseases using AI.
    """)

# -----------------------------
# Disease Recognition Page
# -----------------------------
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Upload a leaf image:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                result_index = model_prediction(uploaded_file)

                if result_index is not None:
                    class_names = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                    ]

                    st.success(f"Model Prediction: **{class_names[result_index]}** 🌿")
