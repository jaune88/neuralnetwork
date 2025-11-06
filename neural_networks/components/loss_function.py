'''Notel: this is doing cross entropy
cost = average of loss functions (only 1 number)
'''

# builtin

# external
import numpy as np

# internal

# superclass / interface for loss functions
class LossFunction():
    # basically, returns the base gradient of whatever loss function you're using
    def get_training_loss(self, response: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        pass
    
    # this actually just returns like accuracy or something
    def get_test_loss(self, response: np.ndarray, prediction: np.ndarray) -> float:
        pass
    

class CrossEntropy(LossFunction):
    def get_training_loss(self, response: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        #this only needs to derivative for backpropagation
        N = prediction.shape[0]
        py = prediction[np.arrange(N), response]
        log_likelihood = -np.log(np.clip(py, 1e-12, 1 - 1e-12))
        raise Exception("TODO: Not yet implemented")
    
    def get_test_loss(self, response: np.ndarray, prediction: np.ndarray) -> float:
        # this one returns the actual value
        N = prediction.shape[0]
        py = prediction[np.arrange(N), response]
        log_likelihood = -np.log(np.clip(py, 1e-12, 1 - 1e-12))
        return np.mean(log_likelihood)