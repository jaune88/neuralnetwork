#builtin
from pathlib import Path

#external
from PIL import Image
import numpy as np

#internal
from network import NeuralNetwork
from components.loss_function import CrossEntropy  # or any LossFunction subclass

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'input_1_1_1.jpg'

def load_image(path: Path) -> np.array:
    img = Image.open(path)
    array = np.array(img)
    normalize = array / 255.0
    flattened_array = normalize.flatten().reshape(1,-1)
    return(flattened_array)

def main():
    x = load_image(DATA_PATH)   # shape: (1, input_size)
    input_size = x.shape[1]

    loss_fn = CrossEntropy() 
    net = NeuralNetwork(
        dimensions=[input_size, 128, 10],
        learning_rate=0.01,
        loss_function=loss_fn,
    )

    probs = net.predict(x)                   
    
    predicted_digit = int(np.argmax(probs, axis=1)[0])

    print("Softmax probabilities:", probs)
    print("Predicted digit:", predicted_digit)

if __name__ == "__main__":
    main()

    
'''
1. forward in layer class
2. within layer calss
3. save weights
4. backpropagation
'''