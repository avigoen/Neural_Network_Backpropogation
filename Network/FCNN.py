import numpy as np

from Utils.network import first_class, get_all_classes
from Layers import Dense, Relu, Softmax
from constants import FLOAT_DTYPE
from .base import BaseNetwork


class FCNN_Network(BaseNetwork):
    def __init__(self) -> None:
        super().__init__()

    def add(self, layer):
        """
        Add layers to the network
        """
        reversed_layers = self.reverse_layers
        latest_dense = first_class(reversed_layers, Dense)
        if layer.__class__ == Relu:
            latest_dense.activation = 'relu'

        if layer.__class__ == Softmax:
            latest_dense.activation = 'softmax'

        self.add_layer(layer)

    def _init_weights(self, input_shape):
        """
        Initialize the model parameters
        """
        np.random.seed(99)

        dense_layers = list(get_all_classes(self.network, Dense))

        for idx, layer in enumerate(dense_layers):
            input_dim = input_shape[1] if idx == 0 else dense_layers[idx - 1].neurons
            output_dim = layer.neurons
            activation = layer.activation

            self.architecture.append(
                {'input_dim': input_dim, 'output_dim': output_dim})

            init_weight_fn = self.xavier_init if activation == 'relu' else self.he_init
            init_weight = FLOAT_DTYPE(init_weight_fn((input_dim, output_dim)))

            layer.weights = init_weight
            layer.bias = FLOAT_DTYPE(np.zeros((1, output_dim)))

        return self

    def compile(self, input_shape):
        """
        Compile the model to initialise the weights and layers
        """

        self._init_weights(input_shape)