import numpy as np
from Layers.properties import ReduceOverfitProperties


class DropoutLayer(ReduceOverfitProperties):
    def __init__(self, prob) -> None:
        super().__init__()
        self.prob = prob
        self.scale = 1 if prob == 1 else 1 / (1 - prob)

    def forward(self):
        '''
        Forward function of the Dropout layer: it create a random mask for every input
            in the batch and set to zero the chosen values. Other pixels are scaled
            with the scale variable.
        Parameters :
            inpt : array of shape (batch, w, h, c), input of the layer
        '''

        self.rnd = np.random.uniform(
            low=0., high=1., size=self.input.shape) > self.prob
        return self.rnd * self.input * self.scale

    def backward(self, delta=None):
        '''
        Backward function of the Dropout layer: given the same mask as the layer
            it backprogates delta only to those pixel which values has not been set to zero
            in the forward.
        Parameters :
            delta : array of shape (batch, w, h, c), default value is None.
                If given, is the global delta to be backpropagated
        '''
        d = np.ones(shape=self.input.shape)
        d = self.rnd * delta * self.scale
        return d.copy()
