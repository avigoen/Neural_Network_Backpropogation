import numpy as np

from Utils.network import get_all_classes
from Layers import Conv2D
from constants import FLOAT_DTYPE
from .base import BaseNetwork


class CNN_Network(BaseNetwork):
    def __init__(self) -> None:
        super().__init__()

    def add(self, layer):
        """
        Add layers to the network
        """
        self.add_layer(layer)

    def _init_weights(self, input_shape):
        """
        Initialize the model parameters
        """
        np.random.seed(99)

        c = input_shape[3]

        conv_layers = get_all_classes(self.network, Conv2D)
        for layer in conv_layers:
            scale = np.sqrt(
                2. / (layer.size[0] * layer.size[1] * c), dtype=FLOAT_DTYPE)
            layer.weights = FLOAT_DTYPE(np.random.normal(loc=scale, scale=1., size=(
                layer.size[0], layer.size[1], c, layer.filters)))

            layer.bias = np.zeros(shape=(layer.filters, ), dtype=FLOAT_DTYPE)

        return self

    def compile(self, input_shape):
        """
        Compile the model to initialise the weights and layers
        """

        self._init_weights(input_shape)
