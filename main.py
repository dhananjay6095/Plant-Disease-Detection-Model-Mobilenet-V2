import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
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
