'''Note: this is where z = f(z) happens
'''

# builtin

# external
import numpy as np

# internal

# superclass / interface for activation functions
class ActivationFunction():
    def apply(self, data: np.ndarray) -> np.ndarray:
        pass

    # note that you need to store the input data to calculate the elementwise gradient in most cases
    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass

# a type fo activation for hidden layers. updates the gradient and send it back to the previous layer
class ReLU(ActivationFunction):
    def apply(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(0, data)
    #this is for backpropagation
    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        relu_derivative = (data > 0).astype(gradient.dtype)
        return gradient * relu_derivative
    
# this outputs the probability of each possibility
class Softmax(ActivationFunction):
    def apply(self, data: np.ndarray) -> np.ndarray:
        e_x = np.exp(data - np.max(data, axis = 1, keepdims=True))
        return e_x / np.sum(e_x, axis = 1, keepdims=True)

    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return gradient