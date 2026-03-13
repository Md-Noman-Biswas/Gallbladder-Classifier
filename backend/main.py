import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from transformers import TFViTModel, ViTConfig

# ──────────────────────────────────────────────────────────────
# 1. VIT BASE MODEL
# ──────────────────────────────────────────────────────────────
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

print("Loading ViT base weights...")
try:
    vit_base_for_loading = TFViTModel.from_pretrained(VIT_MODEL_NAME, from_pt=False)
    print("OK: Native TF ViT weights loaded")
except Exception:
    try:
        vit_base_for_loading = TFViTModel.from_pretrained(VIT_MODEL_NAME, from_pt=True)
        print("OK: PyTorch -> TF ViT weights loaded")
    except Exception:
        config = ViTConfig.from_pretrained(VIT_MODEL_NAME)
        vit_base_for_loading = TFViTModel(config)
        dummy = tf.zeros((1, 3, 224, 224))
        _ = vit_base_for_loading(pixel_values=dummy, training=False)
        print("WARN:  ViT created with random weights")

def vit_forward(x):
    x = tf.transpose(x, [0, 3, 1, 2])
    outputs = vit_base_for_loading(pixel_values=x, training=False)
    return outputs.last_hidden_state[:, 0, :]

# ──────────────────────────────────────────────────────────────
# 2. CUSTOM KERAS LAYERS
# ──────────────────────────────────────────────────────────────
@register_keras_serializable(package='Custom', name='ViTPreprocessLayer')
class ViTPreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32, shape=[1,1,1,3])
        self.std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32, shape=[1,1,1,3])

    def call(self, images):
        return (images - self.mean) / self.std

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package='Custom', name='ViTFeatureExtractor')
class ViTFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, vit_model=None, **kwargs):
        super().__init__(**kwargs)
        self.vit_model = vit_model if vit_model is not None else vit_base_for_loading

    def call(self, inputs, training=False):
        x   = tf.transpose(inputs, [0, 3, 1, 2])
        out = self.vit_model(pixel_values=x, training=training)
        return out.last_hidden_state[:, 0, :]

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    'ViTPreprocessLayer': ViTPreprocessLayer,
    'ViTFeatureExtractor': ViTFeatureExtractor,
    'vit_forward': vit_forward,
}

# ──────────────────────────────────────────────────────────────
# 3. LOAD MODELS (.h5 format)
# ──────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
GAN_PATH   = os.path.join(MODEL_DIR, "generator_best.h5")
CLF_PATH   = os.path.join(MODEL_DIR, "classifier_vit_best_joint.h5")

print("Loading GAN generator...")
generator  = load_model(GAN_PATH,  custom_objects=CUSTOM_OBJECTS, compile=False)
print("OK: Generator loaded")

print("Loading ViT classifier...")
classifier = load_model(CLF_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
print("OK: Classifier loaded")

# ──────────────────────────────────────────────────────────────
# 4. CLASS NAMES  (sorted alphabetically - matches training)
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "1Gallstones",
    "2Abdomen and retroperitoneum",
    "3cholecystitis",
    "4Membranous and gangrenous cholecystitis",
    "5Perforation",
    "6Polyps and cholesterol crystals",
    "7Adenomyomatosis",
    "8Carcinoma",
    "9Various causes of gallbladder wall thickening",
]

# ──────────────────────────────────────────────────────────────
# 5. IMAGE HELPERS
# ──────────────────────────────────────────────────────────────
IMG_SIZE = 224

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype="float32") / 255.0

def array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ──────────────────────────────────────────────────────────────
# 6. FASTAPI APP
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="Gallbladder Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Gallbladder Classifier API is running"}

@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, etc.)")

    image_bytes = await file.read()

    arr       = preprocess_image(image_bytes)
    input_b64 = array_to_base64(arr)

    batch        = arr[np.newaxis, ...]
    denoised_arr = generator.predict(batch, verbose=0)[0]
    denoised_arr = np.clip(denoised_arr, 0, 1).astype("float32")
    denoised_b64 = array_to_base64(denoised_arr)

    probs      = classifier.predict(denoised_arr[np.newaxis, ...], verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    all_probs = [
        {"class": CLASS_NAMES[i], "probability": float(probs[i])}
        for i in np.argsort(probs)[::-1]
    ]

    return {
        "prediction":     pred_class,
        "confidence":     confidence,
        "probabilities":  all_probs,
        "input_image":    input_b64,
        "denoised_image": denoised_b64,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)