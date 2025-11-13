from typing import List
from PIL import Image

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

def preprocess_canvas(arr: np.ndarray) -> np.ndarray:
    """
    Take a 64x64 binary array from the canvas (values 0 or 1),
    crop to the drawn digit, resize back to 64x64, and return
    a float32 array in [0, 1] ready for the network.
    """
    # Safety: ensure 2D
    arr = np.array(arr, dtype=np.float32)
    if arr.shape != (64, 64):
        raise ValueError(f"Expected 64x64 pixels, got {arr.shape}")

    # If nothing is drawn, just return all zeros
    if arr.max() == 0:
        return arr  # all zeros

    # Find bounding box of drawn pixels
    ys, xs = np.where(arr > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Add a small margin so we don't crop too tight
    margin = 2
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, arr.shape[0] - 1)
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, arr.shape[1] - 1)

    cropped = arr[y_min:y_max+1, x_min:x_max+1]

    # Convert to a PIL image for smooth resizing
    img = Image.fromarray((cropped * 255).astype(np.uint8))
    img = img.resize((64, 64), Image.BILINEAR)

    # Back to numpy, normalize to [0, 1]
    norm = np.array(img, dtype=np.float32) / 255.0
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
