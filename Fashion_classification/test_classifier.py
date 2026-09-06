import os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from tensorflow import keras

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

def test_fashion_image(image_filename, model_filename="Fashion_classification_1.keras"):
    """
    Test function to load an image from the 'img' folder (or Fashion_classification directory),
    plug it into the trained model without retraining, and return the classification output.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check img folder first, fallback to base_dir
    img_folder = os.path.join(base_dir, "img")
    image_path = os.path.join(img_folder, image_filename)
    if not os.path.exists(image_path):
        image_path = os.path.join(base_dir, image_filename)
        
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    model_path = os.path.join(base_dir, model_filename)
    if not os.path.exists(model_path):
        # Fallback to cnn model if exists
        model_path = os.path.join(base_dir, "Fashion_classification_cnn.keras")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
        
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"Loading image from: {image_path}")
    pil_img = Image.open(image_path)
    
    # Prepare image
    img_array = prepare_image(pil_img)
    
    # Reshape for model input (CNN or MLP)
    input_shape = model.input_shape
    if len(input_shape) == 4:
        # CNN expects (batch, 28, 28, 1)
        X_input = img_array.reshape(1, 28, 28, 1)
    else:
        # MLP expects (batch, 28, 28) or (batch, 784)
        X_input = img_array.reshape(1, 28, 28)
        
    preds = model.predict(X_input, verbose=0)[0]
    pred_idx = preds.argmax()
    predicted_class = class_names[pred_idx]
    confidence = preds[pred_idx]
    
    print("----------------------------------------")
    print(f"Predicted Fashion Item : {predicted_class}")
    print(f"Confidence             : {confidence:.2%}")
    print("----------------------------------------")
    
    return predicted_class, confidence, preds

if __name__ == "__main__":
    # Example test call (e.g. trouser.jpg or any image in the directory)
    test_fashion_image("trouser.jpg")
