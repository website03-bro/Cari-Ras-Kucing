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

# Warna tetap (tema light)
primary_color = "#f4a300"
background_color = "#ffffff"
text_color = "#000000"
card_color = "#fff6e5"

# Styling CSS
st.markdown(
    f"""
    <style>
    body {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #e69500;
        transform: scale(1.05);
    }}
    .stFileUploader {{
        background-color: {card_color};
        padding: 1em;
        border-radius: 10px;
    }}
    hr {{
        border: 1px solid {primary_color};
    }}
    .logo-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        pointer-events: none;
        user-select: none;
    }}
    img {{
        pointer-events: none;
        user-select: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
MODEL_PATH = "model/MobileNetV2_9150.h5"

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

# ========== Tampilan UI ==========

# Logo
logo = Image.open("Logo/logo web HD.png")
st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
st.image(logo, width=200)
st.markdown("</div>", unsafe_allow_html=True)

# Judul aplikasi
st.markdown(
    f"<h1 style='text-align: center; color: {primary_color};'>Klasifikasi Ras Kucing 🐾</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='text-align: center; font-size: 18px; color: {text_color};'>Unggah gambar kucing favoritmu dan temukan rasnya!</p>",
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

    with st.spinner('⏳ Menganalisis gambar kucing...'):
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

    # Hasil Prediksi
    st.markdown("---")
    st.markdown(
        f"<h2 style='text-align: center; color: {primary_color};'>Hasil Prediksi 🐾</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color:{card_color}; padding:20px; border-radius:10px; text-align:center; box-shadow:0px 0px 10px {primary_color};">
            <h3 style="color:{primary_color};">{predicted_class}</h3>
            <p style="font-size:18px; color:{text_color};">Tingkat Kepercayaan: <strong>{confidence*100:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(confidence))

# Footer
st.markdown(
    f"""
    <hr>
    <div style='text-align: center; font-size: 14px; color: {text_color};'>
        © 2025 - Klasifikasi Ras Kucing dengan AI | Made with ❤️ by Boy
    </div>
    """,
    unsafe_allow_html=True
)
