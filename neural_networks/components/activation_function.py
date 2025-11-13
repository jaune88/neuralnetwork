'''Note: this is where z = f(z) happens
'''
#external
import numpy as np


class ActivationFunction:
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Forward: take z and return a."""
        raise NotImplementedError("apply() must be implemented in subclasses")

    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Backward: given z and dC/da, return dC/dz.
        """
        raise NotImplementedError("apply_gradient() must be implemented in subclasses")


class ReLU(ActivationFunction):
    def apply(self, data: np.ndarray) -> np.ndarray:
        # elementwise max(0, z)
        return np.maximum(0, data)

    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        data: z (pre-activation), same shape as gradient
        gradient: dC/da coming from next layer

        ReLU'(z) = 1 if z > 0, else 0.
        dC/dz = dC/da * ReLU'(z) elementwise.
        """
        relu_derivative = (data > 0).astype(gradient.dtype)
        return gradient * relu_derivative


class Softmax(ActivationFunction):
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        data: logits z, shape (N, C)
        returns softmax probabilities, same shape.
        """
        e_x = np.exp(data - np.max(data, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def apply_gradient(self, data: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        With softmax + cross-entropy, the loss already gives us dC/dz,
        so we just pass the gradient through unchanged.
        """
        return gradient