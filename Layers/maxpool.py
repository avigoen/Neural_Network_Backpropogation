from Utils.conv import *
from constants import FLOAT_DTYPE


class MaxPoolLayer:
    def __init__(self, size, stride=None, pad=False) -> None:
        self.size = size
        self._handle_size_stride(size, stride)

        # for padding
        self.pad = pad
        self.pad_left, self.pad_right, self.pad_bottom, self.pad_top = (
            0, 0, 0, 0)

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

    def _build(self, input_shape=None):
        if input_shape is not None:

            if self.pad:
                self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = evaluate_padding(
                    self.input, self.size, self.stride)

    @property
    def padding(self):
        return (self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)

    @property
    def input_shape(self):
        '''
        Get the input shape as (batch, in_w, in_h, in_channels)
        '''
        return self.input.shape

    @property
    def out_shape(self):
        batch, w, h, c = self.input_shape
        out_height = (h + self.pad_left + self.pad_right -
                      self.size[1]) // self.stride[1] + 1
        out_width = (w + self.pad_top + self.pad_bottom -
                     self.size[0]) // self.stride[0] + 1
        out_channels = c
        return (batch, out_width, out_height, out_channels)

    def forward(self):
        '''
        Forward function of the maxpool layer: It slides a kernel over every input image and return
        the maximum value of every sub-window.
        the function _asStride returns a view of the input arrary with shape
        (batch, out_w, out_h , c, kx, ky), where, for every image in the batch we have:
        out_w * out_h * c sub matrixes kx * ky, containing pixel values.

        Parameters
        ----------
        inpt : array-like
            Input batch of images, with shape (batch, input_w, input_h, input_c).

        Returns
        -------
        self
        '''

        kx, ky = self.size
        st1, st2 = self.stride
        _, w, h, _ = self.input_shape

        self.input = self.input.astype(FLOAT_DTYPE)

        if self.pad:
            mat_pad = pad(self.input, self.padding)
        else:
            # If no padding, cut the last raws/columns in every image in the batch
            mat_pad = self.input[:, : (w - kx) // st1 * st1 + kx,
                                 : (h - ky) // st2 * st2 + ky, ...]

        # Return a strided view of the input array, shape: (batch, 1+(w-kx)//st1,1+(h-ky)//st2 ,c, kx, ky)
        view = asStride(mat_pad, self.size, self.stride)

        # final shape (batch, out_w, out_h, c)
        self.z = np.nanmax(view, axis=(4, 5))

        # # New shape for view, to retrieve indexes
        # new_shape = view.shape[:4] + (kx * ky, )

        # self.indexes = np.nanargmax(view.reshape(new_shape), axis=4)

        # # self.indexes = np.unravel_index(self.indexes.ravel(), (kx, ky)) ?
        # try:
        #     self.indexes = np.unravel_index(
        #         self.indexes.ravel(), shape=(kx, ky))
        # except TypeError:  # retro-compatibility for Numpy version older than 1.16
        #     self.indexes = np.unravel_index(
        #         self.indexes.ravel(), dims=(kx, ky))

        return self.z

    def backward(self, delta):
        '''
        Backward function of maxpool layer: it access avery position where in the input image
        there's a chosen maximum and add the correspondent self.delta value.
        Since we work with a 'view' of delta, the same pixel may appear more than one time,
        and an atomic acces to it's value is needed to correctly modifiy it.

        Parameters
        ----------
        delta : array-like
            Global delta to be backpropagated with shape (batch, out_w, out_h, out_c).

        Returns
        ----------
        self
        '''

        # d = np.ones(shape=self.out_shape, dtype=FLOAT_DTYPE)
        delta[:] = delta.astype(FLOAT_DTYPE)

        # Padding delta in order to create another view
        if self.pad:
            mat_pad = pad(delta, self.padding)
        else:
            mat_pad = delta

        # # Create a view of mat_pad, following the padding true or false
        # net_delta_view = asStride(mat_pad, self.size, self.stride)

        # b, w, h, c = self.z.shape

        # # those indexes are usefull to access 'Atomically'(one at a time) every element in net_delta_view
        # for (i, j, k, l), m, o, D in zip(np.ndindex(b, w, h, c), self.indexes[0], self.indexes[1], np.nditer(d)):
        #     net_delta_view[i, j, k, l, m, o] += D

         # Here delta is correctly modified
        if self.pad:
            _, w_pad, h_pad, _ = mat_pad.shape
            delta[:] = mat_pad[:, self.pad_top: w_pad - self.pad_bottom,
                               self.pad_left: h_pad - self.pad_right, :]
        else:
            delta[:] = mat_pad

        return delta
