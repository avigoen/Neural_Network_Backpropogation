import numpy as np
from Layers.properties import ReduceOverfitProperties
from constants import EPSILON

class BatchNormLayer(ReduceOverfitProperties):
    def __init__(self, epsilon=EPSILON) -> None:
        super().__init__()
        self.eps = epsilon

    def _init_scales(self):
        self.weights = np.random.uniform(
            low=0., high=1., size=self.input.shape)

    def _init_bias(self):
        self.bias = np.random.uniform(low=0., high=1., size=self.input.shape)

    def forward(self):
        '''
        Forward function of the BatchNormalization layer. It computes the output of
        the layer, the formula is :
                        output = scale * input_norm + bias
        Where input_norm is:
                        input_norm = (input - mean) / sqrt(var + epsil)
        where mean and var are the mean and the variance of the input batch of
        images computed over the first axis (batch)
        Parameters:
            inpt  : numpy array, batch of input images in the format (batch, w, h, c)
            epsil : float, used to avoi division by zero when computing 1. / var
        '''

        self._init_scales()
        self._init_bias()

        # Copy input, compute mean and inverse variance with respect the batch axis
        mean = self.input.mean(axis=0)  # Shape = (w, h, c)
        var = 1. / np.sqrt((self.input.var(axis=0)) +
                           self.eps)  # shape = (w, h, c)
        # epsil is used to avoid divisions by zero

        # Compute the normalized input
        self.output = (self.input - mean) * var  # shape (batch, w, h, c)

        # Output = scale * x_norm + bias
        if self.weights is not None:
            self.output *= self.weights  # Multiplication for scales

        if self.bias is not None:
            self.output += self.bias  # Add bias

        # output_shape = (batch, w, h, c)
        return self.output

    def backward(self, delta=None):
        '''
        BackPropagation function of the BatchNormalization layer. Every formula is a derivative
        computed by chain rules: dbeta = derivative of output w.r.t. bias, dgamma = derivative of
        output w.r.t. scales etc...
        Parameters:
            delta : the global error to be backpropagated, its shape should be the same
            as the input of the forward function (batch, w, h ,c)
        '''

        inp_mean = self.input.mean(axis=0)
        inp_var = 1. / np.sqrt((self.input.var(axis=0)) +
                               self.eps)
        invN = 1. / np.prod(inp_mean.shape)

        # Those are the explicit computation of every derivative involved in BackPropagation
        # of the batchNorm layer, where dbeta = dout / dbeta, dgamma = dout / dgamma etc...

        self.delta = np.random.uniform(
            low=0., high=100., size=self.input.shape)

        self.db = self.delta.sum(axis=0)                    # dbeta
        self.dW = (self.delta * self.output).sum(axis=0)  # dgamma

        # self.delta = dx_norm from now on
        self.delta *= self.weights

        mean_delta = (self.delta * (-inp_var)).mean(axis=0)    # dmu

        var_delta = ((self.delta * (self.input - inp_mean)).sum(axis=0) *
                     (-.5 * inp_var * inp_var * inp_var))    # dvar

        # Here, delta is the derivative of the output w.r.t. input
        self.delta = (self.delta * inp_var +
                      var_delta * 2 * (self.input - inp_mean) * invN +
                      mean_delta * invN)

        if delta is not None:
            delta[:] += self.delta

        return delta if delta is not None else self.delta
