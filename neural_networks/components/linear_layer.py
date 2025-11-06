'''Note: z = Wa + b with activation so it turns into a(next) = f(Wa + b)
'''

# builtin

# external
import numpy as np

# internal
from components.activation_function import ActivationFunction

class LinearLayer():

    def __init__(
            self, 
            input_size: int, 
            output_size: int,
            activation: ActivationFunction,
            weights: np.ndarray = None, 
            biases: np.ndarray = None
    ):
        # constructor in Python
        # initialize weights, biases or set them to the load values (if not null)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation 

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(input_size, output_size)
        
        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.random.normal(loc = 0, scale = np.sqrt(2/input_size), size = (1, output_size))
    
# if there's weird results, scale it down
    
    #for backpropagation
    '''self.last_x = None
    self.last_z = None'''

    def forward(self, input: np.ndarray) -> np.ndarray:
        # don't forget about activation
        #input dimensions = batch size x input size, returns batch size x output size
        z = input @ self.weights + self.biases
        return self.activation.apply(z)

 #ignore for now   
    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:

        self.__update_weights(gradient, learning_rate)

        self.__update_biases(gradient, learning_rate)

        # update gradient with Jacobian of this layer to pass back and return
        raise Exception("TODO: Not yet implemented")
    
    def __update_weights(self, gradient: np.ndarray, learning_rate: float):
        raise Exception("TODO: Not yet implemented")
    
    def __update_biases(self, gradient: np.ndarray, learning_rate: float):
        raise Exception("TODO: Not yet implemented")
    
    # return in order of (weight, bias)
    def save_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        raise Exception("TODO: Not yet implemented")