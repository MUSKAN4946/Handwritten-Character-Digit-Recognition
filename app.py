from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf

from src.labels import CLASS_NAMES

st.set_page_config(page_title="Handwritten Recognition", page_icon="✍️", layout="centered")
st.title("✍️ Handwritten Character & Digit Recognition")
st.write("Upload one handwritten digit (0–9) or uppercase letter (A–Z).")

MODEL_PATH = Path(__file__).parent / "models" / "handwritten_cnn.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def prepare_image(uploaded_image):
    """Convert a single-character image into the model's 28x28 input format."""
    image = Image.open(uploaded_image).convert("L")
    image = ImageOps.autocontrast(image)
    arr = np.asarray(image).astype("uint8")

    # Make the character white on a black background, as in MNIST/EMNIST.
    if arr.mean() > 127:
        arr = 255 - arr

    # Remove very faint background noise and resize while preserving aspect ratio.
    arr[arr < 35] = 0
    image = Image.fromarray(arr)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.25))

    x_arr = np.asarray(canvas).astype("float32") / 255.0
    return x_arr.reshape(1, 28, 28, 1), canvas

if not MODEL_PATH.exists():
    st.warning("Trained model not found.")
    st.markdown(
        "First run `notebooks/Handwritten_Character_Digit_Recognition.ipynb` "
        "in Google Colab, download `handwritten_cnn.keras`, and place it inside the `models` folder."
    )
    st.stop()

model = load_model()
uploaded = st.file_uploader("Choose a PNG, JPG or JPEG image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        x, preview = prepare_image(uploaded)
        st.image(preview, caption="Preprocessed 28 × 28 image", width=180)
        probabilities = model.predict(x, verbose=0)[0]
        best = int(np.argmax(probabilities))
        st.success(f"Prediction: **{CLASS_NAMES[best]}**")
        st.metric("Model confidence", f"{probabilities[best] * 100:.2f}%")

        top_ids = np.argsort(probabilities)[-5:][::-1]
        st.subheader("Top 5 predictions")
        for idx in top_ids:
            st.write(f"**{CLASS_NAMES[int(idx)]}** — {probabilities[idx] * 100:.2f}%")
            st.progress(float(probabilities[idx]))
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.caption("Educational demonstration. Predictions may be incorrect, especially for unclear or unusually written characters.")
