from typing import List
from PIL import Image, ImageFilter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from network import NeuralNetwork
from components.loss_function import CrossEntropy

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Predict64(BaseModel):
    pixels: List[List[float]]  # 64x64
        

MODEL_PATH = "trained_network.json"
_loss_fn = CrossEntropy()
_net = NeuralNetwork.load_network(MODEL_PATH, loss_function=_loss_fn)

"""
def preprocess_canvas(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    if arr.shape != (64, 64):
        raise ValueError(f"Expected 64x64 pixels, got {arr.shape}")

    # If nothing is drawn, just return all zeros
    if arr.max() == 0:
        return arr

    # Clip to [0,1] in case anything weird happens
    arr = np.clip(arr, 0.0, 1.0)

    # OPTIONAL: lightly blur so blocky pixels look more MNIST-ish
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))  # try 0.5–1.2

    arr_smooth = np.array(img, dtype=np.float32) / 255.0

    return arr_smooth
"""

def preprocess_canvas(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)

    if arr.shape != (64, 64):
        raise ValueError(f"Expected 64x64 pixels, got {arr.shape}")

    # If nothing is drawn, return zeros
    if arr.max() == 0:
        return arr

    # values are currently 0, 0.7, 1 -> clip and normalize
    arr = np.clip(arr, 0.0, 1.0)

    # OPTIONAL: small blur to make blocky squares look like strokes
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))  # tweak 0.5–1.0

    norm = np.array(img, dtype=np.float32) / 255.0

    # stretch contrast so strokes really stand out
    mn, mx = norm.min(), norm.max()
    norm = norm - mn
    if mx - mn > 1e-6:
        norm = norm / (mx - mn)

    return norm

@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Predict64):
    try:
        arr = np.array(payload.pixels, dtype=np.float32)

        # preprocess to center + resize etc.
        arr_proc = preprocess_canvas(arr)

        # debug AFTER preprocessing
        print(
            "DEBUG input (proc): min", arr_proc.min(),
            "max", arr_proc.max(),
            "sum", arr_proc.sum()
        )

        X = arr_proc.flatten().reshape(1, -1)
        probs = _net.predict(X)

        print("DEBUG probs:", probs[0])

        pred = int(np.argmax(probs, axis=1)[0])
        return {"predicted_digit": pred, "probs": probs[0].tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
