# builtin

# external
import numpy as np

# internal


# superclass / interface for loss functions
class LossFunction:
    # returns the base gradient of whatever loss function you're using
    def get_training_loss(self, response: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # returns a scalar evaluation of the loss (e.g. mean cross-entropy)
    def get_test_loss(self, response: np.ndarray, prediction: np.ndarray) -> float:
        raise NotImplementedError
    

class CrossEntropy(LossFunction):
    def get_training_loss(self, response: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        pred = np.atleast_2d(prediction)
        N = pred.shape[0]

        one_hot = np.zeros_like(pred)
        one_hot[np.arange(N), response] = 1

        grad = (pred - one_hot) / N
        return grad
    
    def get_test_loss(self, response: np.ndarray, prediction: np.ndarray) -> float:
        """Mean cross-entropy value (no gradient)."""
        pred = np.atleast_2d(prediction)
        N = pred.shape[0]
        py = pred[np.arange(N), response]
        log_likelihood = -np.log(np.clip(py, 1e-12, 1 - 1e-12))
        return float(np.mean(log_likelihood))