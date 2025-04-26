import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Path model di dalam folder repository GitHub
MODEL_PATH = "model/MobileNetV2_9150.h5"

# Cache model supaya tidak memuat ulang terus-menerus
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Label ras kucing
CLASS_NAMES = [
    'American Shorthair', 'Bengal', 'Bombay', 'British Shorthair',
    'Himalayan', 'Maine Coon', 'Manx', 'Persian', 'Ragdoll',
    'Russian Blue', 'Scottish Fold', 'Siamese', 'Sphynx'
]

# ================== Tampilan UI ==================

# Atur halaman
st.set_page_config(
    page_title="Klasifikasi Ras Kucing",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="auto"
)

# Menampilkan logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/USERNAME/REPO/main/path/logo.png" alt="Logo" width="200"/>
    </div>
    """,
    unsafe_allow_html=True
)

# Judul aplikasi
st.markdown(
    "<h1 style='text-align: center; color: #ff69b4;'>Klasifikasi Ras Kucing 🐾</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center; font-size: 18px;'>Unggah gambar kucing favoritmu dan temukan rasnya!</p>", unsafe_allow_html=True)

# Upload gambar
st.markdown("---")
uploaded_file = st.file_uploader("Pilih gambar kucing...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Hasil Prediksi
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center; color: #6a5acd;'>Hasil Prediksi 🐾</h2>",
        unsafe_allow_html=True
    )
    st.success(f"**Ras Kucing:** {predicted_class}")
    st.info(f"**Tingkat Kepercayaan:** {confidence:.2%}")

# Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center;'>
        <small>© 2025 - Klasifikasi Ras Kucing dengan AI | Made with ❤️ by Boy</small>
    </div>
    """,
    unsafe_allow_html=True
)
