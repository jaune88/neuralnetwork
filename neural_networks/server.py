from typing import List

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


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Predict64):
    try:
        arr = np.array(payload.pixels, dtype=np.float32)

        # Helpful sanity check
        if arr.shape != (64, 64):
            raise ValueError(f"Expected 64x64 pixels, got {arr.shape}")

        X = arr.flatten().reshape(1, -1)
        probs = _net.predict(X)
        pred = int(np.argmax(probs, axis=1)[0])
        return {"predicted_digit": pred, "probs": probs[0].tolist()}
    except Exception as e:
        # This message will now show up in the frontend
        raise HTTPException(status_code=500, detail=str(e))
