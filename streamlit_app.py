import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

import config
from gradcam import make_gradcam_heatmap, overlay_heatmap
import os


def list_sample_images(total=100):
    """Return up to `total` image file paths from sample_images.

    The returned list alternates NORMAL and PNEUMONIA samples when possible.
    """
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    if not os.path.exists(sample_dir):
        return []

    image_files = sorted(
        fname for fname in os.listdir(sample_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    normal_images = [
        os.path.join(sample_dir, fname)
        for fname in image_files
        if "virus" not in fname.lower()
    ]
    pneumonia_images = [
        os.path.join(sample_dir, fname)
        for fname in image_files
        if "virus" in fname.lower() or "pneumonia" in fname.lower()
    ]

    max_per_class = (total + 1) // 2
    normal_images = normal_images[:max_per_class]
    pneumonia_images = pneumonia_images[:max_per_class]

    images = []
    for i in range(total):
        if i % 2 == 0:
            if i // 2 < len(normal_images):
                images.append((normal_images[i // 2], "NORMAL"))
            elif i // 2 < len(pneumonia_images):
                images.append((pneumonia_images[i // 2], "PNEUMONIA"))
        else:
            if i // 2 < len(pneumonia_images):
                images.append((pneumonia_images[i // 2], "PNEUMONIA"))
            elif i // 2 < len(normal_images):
                images.append((normal_images[i // 2], "NORMAL"))
    return images[:total]


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


def preprocess_pil(img: Image.Image):
    img_rgb = img.convert("RGB")
    original = np.array(img_rgb)
    resized = img_rgb.resize(config.IMG_SIZE)
    arr = np.array(resized) / 255.0
    batch = np.expand_dims(arr, axis=0)
    return batch, original


st.title("Chest X-ray — Pneumonia detector (with Grad-CAM)")
st.write("Upload a chest X-ray image and see the model prediction with Grad-CAM explanation.")

model_path = st.text_input("Model path", config.BEST_MODEL_PATH)
model = load_model(model_path)

if 'selected_path' not in st.session_state:
    st.session_state.selected_path = None

# Sidebar: sample images from sample_images directory
st.sidebar.header("Sample images from sample_images")
sample_images = list_sample_images(total=100)
if sample_images:
    for i, (path, label) in enumerate(sample_images):
        try:
            thumb = Image.open(path).convert("RGB")
            thumb.thumbnail((120, 120))
            st.sidebar.image(thumb, caption=f"{label}", width=120)
            if st.sidebar.button("Use this image", key=f"use_{i}"):
                st.session_state.selected_path = path
                st.experimental_rerun()
        except Exception:
            # ignore missing/corrupt thumbnails
            continue
else:
    st.sidebar.info("No sample images found in the local sample_images directory.")

# Main area: upload or select
uploaded = st.file_uploader("Upload X-ray (jpg, png)", type=["jpg", "jpeg", "png"])

selected_path = st.session_state.selected_path

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.getvalue()))
    source = 'uploaded'
elif selected_path:
    img = Image.open(selected_path)
    source = 'sample'
else:
    img = None
    source = None

if img is not None:
    batch, original = preprocess_pil(img)

    if st.button("Run model"):
        with st.spinner("Running model..."):
            heatmap, prob = make_gradcam_heatmap(batch, model)
            overlay = overlay_heatmap(original, heatmap)

        pred_class = config.CLASS_NAMES[1] if prob >= 0.5 else config.CLASS_NAMES[0]
        confidence = prob if prob >= 0.5 else 1 - prob

        st.subheader(f"Prediction: {pred_class} ({confidence:.1%})")

        col1, col2 = st.columns(2)
        col1.image(original, caption="Original", use_container_width=True)
        col2.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

        # Allow user to save/download the result
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)

        if st.download_button("Download result", data=buf, file_name="gradcam_result.png", mime="image/png"):
            st.success("Result downloaded")
else:
    st.info("Select a sample image from the sidebar or upload your own, then click 'Run model'.")
