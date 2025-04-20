import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Path model di dalam folder repository GitHub
MODEL_PATH = "model/MobileNetV2_9150.h5"

# Cache model agar tidak dimuat ulang terus-menerus
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Label ras kucing
CLASS_NAMES = ['American Shorthair', 'Bengal', 'Bombay', 'British Shorthair',
               'Himalayan', 'Maine Coon', 'Manx', 'Persian', 'Ragdoll',
               'Russian Blue', 'Scottish Fold', 'Siamese', 'Sphynx']

# UI Streamlit
st.title("Klasifikasi Ras Kucing 🐱")
st.write("Unggah gambar kucing untuk mengetahui rasnya.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar kucing...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))  # Ukuran sesuai model
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Output hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(f"**Ras Kucing**: {predicted_class}")
    st.write(f"**Kepercayaan**: {confidence:.2%}")
