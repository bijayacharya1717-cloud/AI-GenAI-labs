import os
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from scipy import ndimage
from tensorflow import keras

st.set_page_config(page_title="Fashion Item Classifier")
st.title("Fashion Item Classifier")

# Fashion MNIST class labels
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Set model path to the model in the Fashion_classification folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Fashion_classification_1.keras")

if not os.path.exists(MODEL_PATH):
    st.error(f"No model file found at `{MODEL_PATH}`")
    st.info("Please ensure `Fashion_classification_1.keras` is in the `Fashion_classification` directory.")
    st.stop()

model = keras.models.load_model(MODEL_PATH)


def prepare_image(pil_image):
    """Turn a photo of a clothing item into the 28x28 input the model expects."""
    gray = pil_image.convert("L")
    a = np.asarray(gray, dtype=np.float32)
    big = max(gray.size)

    def ink(radius):
        bg = np.asarray(
            gray.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
        )
        return np.clip(bg - a, 0, None)

    rough = ink(big / 12.0)
    if rough.max() == 0:
        return np.zeros((28, 28), np.float32)
    rough_mask = rough > 0.4 * rough.max()
    if not rough_mask.any():
        return np.zeros((28, 28), np.float32)
    stroke = 2.0 * np.percentile(
        ndimage.distance_transform_edt(rough_mask)[rough_mask], 90
    )
    a = ink(float(np.clip(3.0 * stroke, big / 60.0, big / 3.0)))

    if a.max() == 0:
        return np.zeros((28, 28), np.float32)

    mask = a > 0.35 * a.max()
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    if not mask.any():
        return np.zeros((28, 28), np.float32)

    ys, xs = np.where(mask)
    a = a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    a = a / a.max() * 255.0

    h, w = a.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = np.asarray(
        Image.fromarray(a.astype(np.uint8)).resize((nw, nh), Image.LANCZOS),
        dtype=np.float32,
    )

    out = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    out[top : top + nh, left : left + nw] = small

    total = out.sum()
    if total > 0:
        idx = np.arange(28)
        dy = int(round(13.5 - out.sum(1) @ idx / total))
        dx = int(round(13.5 - out.sum(0) @ idx / total))
        shifted = np.zeros_like(out)
        y0, y1 = max(0, dy), 28 + min(0, dy)
        x0, x1 = max(0, dx), 28 + min(0, dx)
        shifted[y0:y1, x0:x1] = out[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        out = shifted

    return out / max(out.max(), 1e-6)


if "upload_round" not in st.session_state:
    st.session_state.upload_round = 0

st.caption("Upload a photo of a fashion item (e.g. T-shirt, shoe, bag).")
image_file = st.file_uploader(
    "Upload fashion image",
    type=["png", "jpg", "jpeg"],
    key=f"fashion_image_{st.session_state.upload_round}",
)

if image_file is not None:
    img_array = prepare_image(Image.open(image_file))

    left, right = st.columns(2)

    with left:
        st.image(Image.open(image_file), caption="Your image", width=250)

    with right:
        y_prob = model.predict(img_array.reshape(1, 28, 28), verbose=0)[0]
        y_pred_idx = y_prob.argmax()
        predicted_item = class_names[y_pred_idx]

        st.markdown(f"# Predicted: {predicted_item}")
        st.progress(
            float(y_prob[y_pred_idx]), text=f"{y_prob[y_pred_idx]:.1%} confident"
        )

    st.bar_chart({name: prob for name, prob in zip(class_names, y_prob)})

    with st.expander("What the model sees (28x28)"):
        st.caption("This should be a white clothing item on a black background.")
        st.image(img_array, width=140, clamp=True)

    if st.button("Upload another image", type="primary"):
        st.session_state.upload_round += 1
        st.rerun()
