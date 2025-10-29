from PIL import Image
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'input_1_1_1.jpg'

if __name__ == "__main__":
    img = Image.open(DATA_PATH)

    # Convert the image to a NumPy array
    np_array = np.array(img)

    print(f"Shape of the NumPy array: {np_array.shape}")
    print(f"Data type of the NumPy array: {np_array.dtype}")