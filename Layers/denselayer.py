import numpy as np

from Layers.properties import DenseLayerProperties
from Utils.maths import dot_product_batches


class DenseLayer(DenseLayerProperties):
    def __init__(self, neurons):
        super().__init__(neurons)

    def forward(self):
        """
        Single Layer Forward Propagation
        """

        Z_curr = dot_product_batches(self.input, self.weights) + self.bias
        self._z = Z_curr

        return Z_curr

    def backward(self, input):
        """
        Single Layer Backward Propagation
        """

        dW = dot_product_batches(self.input.T, input)
        db = np.sum(input, axis=0, keepdims=True)
        dA = dot_product_batches(input, self.weights)

        self.dW = dW
        self.db = db
        return dA
