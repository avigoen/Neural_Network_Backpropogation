import numpy as np

from constants import FLOAT_DTYPE
from Utils.conv import *
from Layers.properties import ConvLayerProperties


class Conv2d(ConvLayerProperties):
    def __init__(self, size, filters, stride=1, pad=False):
        super().__init__()
        self.filters = filters
        self.channels_out = filters
        self._handle_size_stride(size, stride)

        # Padding
        self.pad = pad
        self.pad_left, self.pad_right, self.pad_bottom, self.pad_top = (
            0, 0, 0, 0)

        # self._build()

    def _build(self):
        '''
        Init layer weights and biases and set the correct
        layer out_shapes.

        Returns
        -------
        self
        '''

        _, w, h, c = self.input_shape

        if self.pad:
            self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = evaluate_padding(
                self.input, self.size, self.stride)

        self.out_w = 1 + (w + self.pad_top + self.pad_bottom -
                          self.size[0]) // self.stride[0]
        self.out_h = 1 + (h + self.pad_left + self.pad_right -
                          self.size[1]) // self.stride[1]

        return self

    @property
    def padding(self):
        return (self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)

    def _handle_size_stride(self, size, stride):
        self.size = size
        if not hasattr(self.size, '__iter__'):
            self.size = (int(self.size), int(self.size))

        if not stride:
            self.stride = size
        else:
            self.stride = stride

        if not hasattr(self.stride, '__iter__'):
            self.stride = (int(self.stride), int(self.stride))

    @property
    def input_shape(self):
        '''
        Get the input shape as (batch, in_w, in_h, in_channels)
        '''
        return self.input.shape

    @property
    def out_shape(self):
        '''
        Get the output shape as (batch, out_w, out_h, out_channels)
        '''
        return (self.input_shape[0], self.out_w, self.out_h, self.channels_out)

    def forward(self):
        '''
        Forward function of the Convolutional Layer: it convolves an image with 'channels_out'
        filters with dimension (kx, ky, channels_in). In doing so, it creates a view of the image
        with shape (batch, out_w, out_h, in_c, kx, ky) in order to perform a single matrix
        multiplication with the reshaped filters array, which shape is (in_c * kx * ky, out_c).

        Parameters
        ----------
        inpt : array-like
            Input batch of images in format (batch, in_w, in_h, in _c)

        copy : bool (default=False)
            If False the activation function modifies its input, if True make a copy instead

        Returns
        -------
        self
        '''
        
        self._build()

        kx, ky = self.size
        sx, sy = self.stride
        _, w, h, _ = self.input_shape
        self.input = self.input.astype(FLOAT_DTYPE)

        # Padding
        if self.pad:
            mat_pad = pad(self.input, self.padding)
        else:
            # If no pad, every image in the batch is cut
            mat_pad = self.input[:, : (w - kx) // sx * sx + kx,
                                 : (h - ky) // sy * sy + ky, ...]

        # Create the view of the array with shape (batch, out_w ,out_h, kx, ky, in_c)
        self.view = asStride(mat_pad, self.size, self.stride)

        # the choice of numpy.einsum is due to reshape of self.view being a copy
        self.z = np.einsum('lmnijk, ijko -> lmno', self.view,
                           self.weights, optimize=True, dtype=FLOAT_DTYPE) + self.bias
        return self.z

    def backward(self, delta):
        '''
        Backward function of the Convolutional layer.
        Source: https://arxiv.org/abs/1603.07285

        Parameters
        ----------
        delta : array-like
            delta array of shape (batch, w, h, c). Global delta to be backpropagated.

        copy : bool (default=False)
            States if the activation function have to return a copy of the input or not.

        Returns
        -------
        self
        '''
        d = np.ones(shape=self.out_shape, dtype=FLOAT_DTYPE)
        delta[:] = delta.astype(FLOAT_DTYPE)

        d *= delta

        self.dW = np.einsum('ijklmn, jkio -> lmno',
                            self.view, d, optimize=True, dtype=FLOAT_DTYPE)
        self.db = d.sum(axis=(0, 1, 2))

        # Rotated weights, as theory suggest
        w_rot = np.rot90(self.weights, 2, axes=(0, 1))

        # Pad and dilate the delta array, then stride it and convolve
        d = dilate_pad(d, self.input_shape, self.out_shape,
                       self.size, self.stride, self.padding)
        delta_view = asStride(d, self.size, self.stride, back=True)
        delta[:] = np.einsum('ijklmn, lmon -> ijko',
                             delta_view, w_rot, optimize=True, dtype=FLOAT_DTYPE)

        return delta
