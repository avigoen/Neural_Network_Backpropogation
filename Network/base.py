from time import time
import numpy as np
from tqdm import tqdm
from Callbacks import EarlyStopping, ReduceLR
from constants import EPSILON

class BaseNetwork:
    def __init__(self) -> None:
        self.network = []  # layers
        self.architecture = []  # mapping input neurons --> output neurons
        self.layers_size = len(self.network)

    @staticmethod
    def generate_weight(shape, stddev):
        return np.random.randn(shape[0], shape[1]) * stddev

    @staticmethod
    def xavier_init(shape):
        stddev = np.sqrt(2 / (shape[0] + shape[1]))
        return BaseNetwork.generate_weight(shape, stddev)

    @staticmethod
    def he_init(shape):
        stddev = np.sqrt(2 / shape[0])
        return BaseNetwork.generate_weight(shape, stddev)

    def add_layer(self, layer):
        """
        Add layers to the network
        """
        self.network.append(layer)
        self.layers_size += 1

    @property
    def reverse_layers(self):
        return self.network[::-1]
    
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

        for layer in tqdm(self.reverse_layers):
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
    
