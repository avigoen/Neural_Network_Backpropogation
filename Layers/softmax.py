import numpy as np

from Layers.properties import ActivationProperties
from constants import FLOAT_DTYPE


class SoftmaxLayer(ActivationProperties):
    def __init__(self):
        super().__init__()

    def forward(self):
        """
        Softmax Activation Function
        """

        exp_scores = np.exp(self.input - np.max(self.input), dtype=FLOAT_DTYPE)
        probs = exp_scores / \
            np.sum(exp_scores, axis=1, keepdims=True, dtype=FLOAT_DTYPE)
        return FLOAT_DTYPE(probs)

    def backward(self, input):
        return input
