import numpy as np

from constants import EPSILON
from Layers.properties import ActivationProperties


class ReluLayer(ActivationProperties):
    def __init__(self):
        super().__init__()

    def forward(self):
        """
        ReLU Activation Function
        """
        return np.maximum(0, self.input)

    def backward(self, dA):
        """
        ReLU Derivative Function
        """
        dZ = np.array(dA, copy=True)
        dZ[self.input <= EPSILON] = 0
        return dZ
