import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/MobileNetV2_9150.h5")
    return model

model = load_model()

# Label ras kucing
class_names = ['American Shorthair', 'Bengal', 'Birman', 'Bombay', 
               'British Shorthair', 'Egyptian Mau', 'Maine Coon', 
               'Persian', 'Ragdoll', 'Russian Blue', 'Siamese', 
               'Sphynx', 'Turkish Angora']

# Judul Aplikasi
st.title("Klasifikasi Ras Kucing 🐱")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar kucing...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))  # Sesuaikan dengan input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### Prediksi: **{predicted_class}**")
