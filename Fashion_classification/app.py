import os
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sidebar model selection (Default to MLP for maximum stability)
st.sidebar.header("Model Selection")
model_options = {}
mlp_path = os.path.join(BASE_DIR, "Fashion_classification_1.keras")
cnn_path = os.path.join(BASE_DIR, "Fashion_classification_cnn.keras")

if os.path.exists(mlp_path):
    model_options["MLP (Dense Model - Stable)"] = mlp_path
if os.path.exists(cnn_path):
    model_options["CNN (Convolutional Model)"] = cnn_path

if not model_options:
    st.error("No trained model files found (`Fashion_classification_1.keras` or `Fashion_classification_cnn.keras`).")
    st.stop()

# Default to MLP if available for robust error-free experience
default_index = 0
for idx, name in enumerate(model_options.keys()):
    if "MLP" in name:
        default_index = idx
        break

selected_model_name = st.sidebar.selectbox("Choose Model Architecture", list(model_options.keys()), index=default_index)
MODEL_PATH = model_options[selected_model_name]

@st.cache_resource
def load_model(path):
    try:
        return keras.models.load_model(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

model = load_model(MODEL_PATH)
if model is None:
    st.stop()

st.sidebar.success(f"Loaded: {selected_model_name}")


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
    try:
        pil_img = Image.open(image_file)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            rgba_img = pil_img.convert("RGBA")
            bg = Image.new("RGB", rgba_img.size, (255, 255, 255))
            bg.paste(rgba_img, (0, 0), rgba_img)
            pil_img = bg
        else:
            pil_img = pil_img.convert("RGB")
        img_array = prepare_image(pil_img)
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
        st.stop()

    left, right = st.columns(2)

    with left:
        st.image(pil_img, caption="Your image", width=250)

    with right:
        try:
            # Robust dimension handling for CNN (4D) vs MLP (3D)
            input_shape = model.input_shape
            if len(input_shape) == 4:
                # CNN expects (batch, height, width, channels)
                X_model = img_array.reshape(1, 28, 28, 1).astype(np.float32)
            else:
                # MLP expects (batch, height, width) or (batch, features)
                X_model = img_array.reshape(1, 28, 28).astype(np.float32)
                
            y_prob = model.predict(X_model, verbose=0)[0]
            y_pred_idx = int(y_prob.argmax())
            predicted_item = class_names[y_pred_idx]

            st.markdown(f"# Predicted: {predicted_item}")
            st.progress(
                float(y_prob[y_pred_idx]), text=f"{y_prob[y_pred_idx]:.1%} confident"
            )
        except Exception as e:
            st.error(f"Prediction error with model `{selected_model_name}`: {e}")
            st.info("Try switching to the MLP (Dense Model) in the sidebar for stable predictions.")
            st.stop()

    st.bar_chart({name: float(prob) for name, prob in zip(class_names, y_prob)})

    with st.expander("What the model sees (28x28)"):
        st.caption("This should be a white clothing item on a black background.")
        st.image(img_array, width=140, clamp=True)

    if st.button("Upload another image", type="primary"):
        st.session_state.upload_round += 1
        st.rerun()
