import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Atur halaman
st.set_page_config(
    page_title="Klasifikasi Ras Kucing",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="auto"
)

# Styling CSS untuk background, tombol, efek hover, file uploader
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
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
        background-color: #fff6e5;
        padding: 1em;
        border-radius: 10px;
    }
    hr {
        border: 1px solid #f4a300;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Path model
MODEL_PATH = "model/MobileNetV2_9150.h5"

# Cache model
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Label kelas
CLASS_NAMES = [
    'American Shorthair', 'Bengal', 'Bombay', 'British Shorthair',
    'Himalayan', 'Maine Coon', 'Manx', 'Persian', 'Ragdoll',
    'Russian Blue', 'Scottish Fold', 'Siamese', 'Sphynx'
]

# ========== Tampilan UI ==========

# Logo
logo = Image.open("Logo/logo web HD.png")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(logo, width=200)
st.markdown("</div>", unsafe_allow_html=True)

# Judul
st.markdown(
    "<h1 style='text-align: center; color: #f4a300;'>Klasifikasi Ras Kucing 🐾</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Unggah gambar kucing favoritmu dan temukan rasnya!</p>",
    unsafe_allow_html=True
)

# Upload gambar
st.markdown("---")
uploaded_file = st.file_uploader("📤 Pilih gambar kucing...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Spinner saat prediksi berjalan
    with st.spinner('⏳ Menganalisis gambar kucing...'):
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

    # Hasil Prediksi
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center; color: #f4a300;'>Hasil Prediksi 🐾</h2>",
        unsafe_allow_html=True
    )

    # Tampilkan hasil dalam card
    st.markdown(
        f"""
        <div style="background-color:#fff6e5; padding:20px; border-radius:10px; text-align:center; box-shadow:0px 0px 10px #f4a300;">
            <h3 style="color:#f4a300;">{predicted_class}</h3>
            <p style="font-size:18px;">Tingkat Kepercayaan:</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(float(confidence))

# Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 14px; color: #666;'>
        © 2025 - Klasifikasi Ras Kucing dengan AI | Made with ❤️ by Boy
    </div>
    """,
    unsafe_allow_html=True
)
