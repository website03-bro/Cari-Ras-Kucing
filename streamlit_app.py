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

# ==== CSS Styling Adaptif ====
st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        background-color: #f4a300 !important;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }

    .stButton>button {
        background-color: #f4a300;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e69500;
        transform: scale(1.05);
    }
    .stFileUploader {
        background-color: var(--secondary-background-color);
        padding: 1em;
        border-radius: 10px;
    }
    hr {
        border: 1px solid #f4a300;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .header-container img {
        max-height: 60px;
        height: auto;
        width: auto;
    }
    img {
        pointer-events: none;
        user-select: none;
    }
    .cat-gallery {
        margin-top: 20px;
    }
    @media (max-width: 768px) {
        h1 {
            font-size: 28px !important;
        }
        .cat-gallery img {
            width: 100% !important;
            height: auto !important;
        }
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

# ==== UI ====

# Logo + Judul
logo = Image.open("Logo/logo web HD.png")
st.markdown("<div class='header-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 6])

with col1:
    st.image(logo, use_container_width=False, width=80)

with col2:
    st.markdown(
        "<h1 style='color: #f4a300; margin: 0;'>Klasifikasi Ras Kucing 🐾</h1>",
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# Subtitle
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

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    with st.spinner('⏳ Menganalisis gambar kucing...'):
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

    # Hasil Deteksi
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center; color: #f4a300;'>Hasil Deteksi</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: var(--secondary-background-color); padding:20px; border-radius:10px; text-align:center; box-shadow:0px 0px 10px #f4a300;">
            <h3 style="color:#f4a300;">{predicted_class}</h3>
            <p style="font-size:18px;">Tingkat Akurasi: <strong>{confidence*100:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(confidence))

# ==== Galeri Gambar Contoh Dipindah ke Bawah ====
st.markdown("---")
st.markdown("### 📸Jenis Ras Kucing", unsafe_allow_html=True)

img_paths = [
    "images/American ShortHair.jpg",
    "images/Bengal.jpg",
    "images/Bombay.jpg",
    "images/British Shorthair.jpg",
    "images/Himalayan.jpg",
    "images/Maine Coon.jpg",
    "images/Manx.jpg",
    "images/persia.jpg",
    "images/ragdoll.jpg",
    "images/russian blue.jpg"
]

st.markdown("<div class='cat-gallery'>", unsafe_allow_html=True)
cols = st.columns(5)
for i, img_path in enumerate(img_paths):
    try:
        with cols[i % 5]:
            st.image(img_path, use_container_width=True, caption=CLASS_NAMES[i])
    except Exception:
        with cols[i % 5]:
            st.warning(f"❌ {CLASS_NAMES[i]}")
            st.text("Gambar tidak ditemukan")
st.markdown("</div>", unsafe_allow_html=True)

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
