# builtin
from __future__ import annotations

# external
import numpy as np

# internal
from components.loss_function import LossFunction
from components.linear_layer import LinearLayer
from components.activation_function import ReLU, Softmax

class NeuralNetwork():
    def __init__(self, dimensions: list[int], learning_rate: float, loss_function: LossFunction):
        # initialize the layers accordingly + setup loss function
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.layers: list[LinearLayer] = []
        
        
        for i in range(len(dimensions) - 1):
            input_size = dimensions[i]
            output_size = dimensions[i + 1]

            # All hidden layers use ReLU, last layer uses Softmax
            if i < len(dimensions) - 2:
                activation = ReLU()
            else:
                activation = Softmax()

            layer = LinearLayer(input_size, output_size, activation)
            self.layers.append(layer)
    
    @classmethod
    def from_save(cls, path: str) -> NeuralNetwork:
        raise Exception("TODO: Not yet implemented")
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, input: np.ndarray, response: np.ndarray) -> np.ndarray:
        raise Exception("TODO: Not yet implemented")
    
    def save_network(self, path: str):
        raise Exception("TODO: Not yet implemented")