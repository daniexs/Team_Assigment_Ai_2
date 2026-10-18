import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_cnn_model.keras", compile=False)

model = load_model()

st.title("Klasifikasi Gambar CIFAR-10 dengan CNN")
st.write("Upload gambar, lalu model akan memprediksi kelasnya.")
st.caption("Kelas yang dikenali: " + ", ".join(CLASS_NAMES))

file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocessing: crop tengah dan resize ke 32x32.
    # Normalisasi 1/255 sudah ada di dalam model, jadi tidak dilakukan di sini.
    img_small = ImageOps.fit(img, (32, 32), Image.LANCZOS)
    x = np.expand_dims(np.array(img_small, dtype="float32"), axis=0)

    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))

    st.success(f"Prediksi: **{CLASS_NAMES[top]}** ({probs[top] * 100:.2f}%)")
    if probs[top] < 0.5:
        st.warning("Keyakinan model rendah. Gambar mungkin bukan salah satu dari 10 kelas di atas.")

    st.subheader("Probabilitas tiap kelas")
    st.bar_chart(pd.DataFrame({"probabilitas": probs}, index=CLASS_NAMES))