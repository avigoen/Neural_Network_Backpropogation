import numpy as np

from tqdm import tqdm
from time import time

from constants import EPSILON, FLOAT_DTYPE
from Callbacks import EarlyStopping, ReduceLR
from Layers import Relu, Softmax, Dense
from Utils.network import first_class, get_all_classes


class Network:
    def __init__(self):
        self.network = []  # layers
        self.architecture = []  # mapping input neurons --> output neurons
        self.params = []  # W, b
        self.layers_size = len(self.network)

    def add(self, layer):
        """
        Add layers to the network
        """
        reversed_layers = self.reverse_layers()
        latest_dense = first_class(reversed_layers, Dense)
        if layer.__class__ == Relu:
            latest_dense.activation = 'relu'

        if layer.__class__ == Softmax:
            latest_dense.activation = 'softmax'

        self.network.append(layer)
        self.layers_size += 1

    def reverse_layers(self):
        return self.network[::-1]

    @staticmethod
    def generate_weight(shape, stddev):
        return np.random.randn(shape[0], shape[1]) * stddev

    @staticmethod
    def xavier_init(shape):
        stddev = np.sqrt(2 / (shape[0] + shape[1]))
        return Network.generate_weight(shape, stddev)

    @staticmethod
    def he_init(shape):
        stddev = np.sqrt(2 / shape[0])
        return Network.generate_weight(shape, stddev)

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

    def _forwardprop(self, data):
        """
        Performs one full forward pass through network
        """
        A_curr = data
        for layer in tqdm(self.network):
            A_prev = A_curr
            layer.input = A_prev
            A_curr = layer.forward()
        return A_curr

    def _backprop(self, predicted, actual):
        """
        Performs one full backward pass through network
        """
        num_samples = len(actual)

        # compute the gradient on predictions
        dscores = predicted
        dscores[np.arange(num_samples).reshape(-1, 1), actual] -= 1
        dscores /= num_samples

        dA_prev = dscores

        for layer in tqdm(self.reverse_layers()):
            dA_curr = dA_prev
            dA_prev = layer.backward(dA_curr)

    def _update(self, lr_reduce):
        """
        Update the model parameters --> lr * gradient
        """
        lr = lr_reduce.get_lr()

        for layer in self.network:
            if layer.weights is not None:
                w = layer.weights
                if layer.dW is None:
                    layer.dW = np.zeros(shape=w.shape)
                w = w - (lr * layer.dW)
                layer.weights = w

            if layer.bias is not None:
                b = layer.bias
                if layer.db is None:
                    layer.db = np.zeros(shape=b.shape)
                b = b - (lr * layer.db)
                layer.bias = b

    def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        return np.mean(np.argmax(predicted, axis=1) == np.argmax(actual, axis=1))

    def _calculate_loss(self, predicted, actual):
        """
        Calculate cross-entropy loss after each iteration
        """
        sample_len = len(actual)
        predicted_one_hot = predicted[np.arange(
            sample_len).reshape(-1, 1), actual]
        correct_logprobs = -np.log(predicted_one_hot + EPSILON)
        no_nan = np.nan_to_num(correct_logprobs, nan=EPSILON)
        data_loss = np.average(no_nan, axis=(0, 1))

        return data_loss

    def _train_batch(self, epoch, x_batch, y_batch):
        """
        Train the model using SGD for batch
        """

        print(f"Epoch {epoch + 1} Forward Propogation Started")
        yhat = self._forwardprop(x_batch)
        print(f"Epoch {epoch + 1} Forward Propogation Finished")

        print(f"Epoch {epoch + 1} Backward Propogation Started")
        self._backprop(predicted=yhat, actual=y_batch)
        print(f"Epoch {epoch + 1} Backward Propogation Finished")

        accuracy = self._get_accuracy(yhat, y_batch)
        loss = self._calculate_loss(yhat, y_batch)

        return accuracy, loss

    def compile(self, input_shape):
        """
        Compile the model to initialise the weights and layers
        """

        self._init_weights(input_shape)

    def train(self, x_train, y_train, epochs):
        """
        Train the model using SGD
        """

        earlyStop = EarlyStopping()
        lr_reduce = ReduceLR(patience=5)

        self.loss = []
        self.accuracy = []

        for i in range(epochs):
            print(
                f"Epoch {i+1} / {epochs} in progress =========================================")
            start_time = time()

            accuracy, loss = self._train_batch(i, x_train, y_train)

            self.accuracy.append(accuracy)
            self.loss.append(loss)

            earlyStop(self.loss[-1])
            lr_reduce(self.loss[-1])

            self._update(lr_reduce)

            end_time = time()
            print(
                f"ACCURACY: {self.accuracy[-1]}, LOSS: {self.loss[-1]}, TIME: {end_time-start_time} sec")
            print(
                f"Epoch {i+1} / {epochs} ends ================================================")

            if lr_reduce.reduce_lr:
                lr_reduce.reset()

            if earlyStop.early_stop:
                print(f"Stopping early at epoch: {i}")
                break
