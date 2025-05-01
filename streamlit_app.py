import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ==== Basic Setup ====
st.set_page_config(
    page_title="Klasifikasi Ras Kucing",
    page_icon=Image.open("Logo/logo web HD.png"),
    layout="centered",
    initial_sidebar_state="auto"
)

# ==== CSS Styling Modern ====
st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        background-color: #f4a300;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }

    body, .main {
        background-color: #fef9f0;
        font-family: 'Segoe UI', sans-serif;
    }

    .header-container {
        background-color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-top: 20px;
        margin-bottom: 30px;
    }

    .header-container img {
        height: 60px;
    }

    .header-container h1 {
        color: #f4a300;
        font-size: 2rem;
        margin: 0;
    }

    .stFileUploader {
        background-color: white;
        border: 2px dashed #f4a300;
        padding: 20px;
        border-radius: 15px;
    }

    .stButton>button {
        background-color: #f4a300;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.8em 2em;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #e69500;
        transform: scale(1.05);
    }

    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(244,163,0,0.25);
        text-align: center;
    }

    .result-card h3 {
        color: #f4a300;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }

    .result-card p {
        font-size: 1.1rem;
        margin: 0;
    }

    footer {
        text-align: center;
        font-size: 0.9rem;
        margin-top: 40px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# ==== Load Model ====
MODEL_PATH = "model/MobileNetV2_9150.h5"

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# ==== Label Kelas ====
CLASS_NAMES = [
    'American Shorthair', 'Bengal', 'Bombay', 'British Shorthair',
    'Himalayan', 'Maine Coon', 'Manx', 'Persian', 'Ragdoll',
    'Russian Blue', 'Scottish Fold', 'Siamese', 'Sphynx'
]

# ==== UI Logo dan Judul ====
st.markdown("<div class='header-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 6])

with col1:
    st.image("Logo/logo web HD.png", width=60)

with col2:
    st.markdown("<h1>Klasifikasi Ras Kucing 🐾</h1>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==== Subtitle ====
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Unggah gambar kucing favoritmu dan temukan rasnya!</p>",
    unsafe_allow_html=True
)

# ==== Upload Gambar ====
st.markdown("---")
uploaded_file = st.file_uploader("📤 Pilih gambar kucing...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    # ==== Preprocessing ====
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # ==== Prediksi ====
    with st.spinner('⏳ Menganalisis gambar kucing...'):
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

    # ==== Hasil Prediksi ====
    st.markdown("---")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3>{predicted_class}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p>Tingkat Kepercayaan: <strong>{confidence*100:.2f}%</strong></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.progress(float(confidence))

# ==== Footer ====
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        © 2025 - Klasifikasi Ras Kucing | Made by Boy
    </div>
    """,
    unsafe_allow_html=True
)
