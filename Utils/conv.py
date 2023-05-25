import numpy as np
from constants import FLOAT_DTYPE


def evaluate_padding(input, size, stride):
    '''
    Compute padding dimensions following keras SAME padding.
    See also:
    https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave
    '''
    _, w, h, _ = input.shape
    # Compute how many Raws are needed to pad the image in the 'w' axis
    if (w % stride[0] == 0):
        pad_w = max(size[0] - stride[0], 0)
    else:
        pad_w = max(size[0] - (w % stride[0]), 0)

    # Compute how many Columns are needed to pad the image in 'h' axis
    if (h % stride[1] == 0):
        pad_h = max(size[1] - stride[1], 0)
    else:
        pad_h = max(size[1] - (h % stride[1]), 0)

    # Number of raws/columns to be added for every directons
    pad_top = pad_w >> 1  # bit shift, integer division by two
    pad_bottom = pad_w - pad_top
    pad_left = pad_h >> 1
    pad_right = pad_h - pad_left

    return pad_top, pad_bottom, pad_left, pad_right


def pad(inpt, padding):
    '''
    Pad every image in a batch with zeros, following keras SAME padding.

    Parameters
    ----------
      inpt : array-like
        input images to pad in the format (batch, in_w, in_h, in_c).

    Returns
    -------
      padded : array-like
        Padded input array, following keras SAME padding format.
    '''
    pad_top, pad_bottom, pad_left, pad_right = padding
    # return the zeros-padded image, in the same format as inpt (batch, in_w + pad_w, in_h + pad_h, in_c)
    return np.pad(inpt, pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                  mode='constant', constant_values=(0., 0.))


def dilate_pad(arr, input_shape, output_shape, size, stride, padding):
    '''
    Dilate input array for backward pass

    reference:
    https://mc.ai/backpropagation-for-convolution-with-strides/

    Parameters
    ----------
      arr : array-like
       input array to be dilated and padded with shape (b, out_w, out_h, out_c)

    Returns
    -------
      dilated : array-like
        The dilated array
    '''

    b, ow, oh, oc = output_shape
    b, w, h, _ = input_shape

    kx, ky = size
    sx, sy = stride
    dx, dy = sx - 1, sy - 1

    final_shape_dilation = (b, ow * sx - dx, oh * sy - dy, oc)

    dilated = np.zeros(shape=final_shape_dilation, dtype=FLOAT_DTYPE)

    dilated[:, ::sx, ::sy, :] = arr

    pad_top, pad_bottom, pad_left, pad_right = padding

    input_pad_w = (pad_top + pad_bottom)
    input_pad_h = (pad_left + pad_right)

    pad_width = (w - kx + input_pad_w) % sx
    pad_height = (h - ky + input_pad_h) % sy

    pad_H_l = ky - pad_left - 1
    pad_H_r = ky - pad_right - 1 + pad_height

    pad_W_t = kx - pad_top - 1
    pad_W_b = kx - pad_bottom - 1 + pad_width

    dilated = np.pad(dilated,
                     pad_width=((0, 0),
                                (pad_W_t, pad_W_b),
                                (pad_H_l, pad_H_r),
                                (0, 0)),
                     mode='constant',
                     constant_values=(0., 0.))

    return dilated


def asStride(arr, size, stride, back=False):
    '''
    _asStride returns a view of the input array such that a kernel of size = (kx,ky)
    is slided over the image with stride = (st1, st2)

    better reference here :
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html

    see also:
    https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy

    Parameters
    ----------
      arr : array-like
        Input batch of images to be convoluted with shape = (b, w, h, c)

    Returns
    -------
      subs : array-view
        View of the input array with shape (batch, out_w, out_h, kx, ky, out_c)
    '''

    batch_stride, s0, s1, s3 = arr.strides
    batch, w, h, c = arr.shape
    kx, ky = size
    st1, st2 = stride

    if back:
        st1 = st2 = 1

    out_w = 1 + (w - kx) // st1
    out_h = 1 + (h - ky) // st2

    # Shape of the final view
    view_shape = (batch, out_w, out_h, c) + (kx, ky)

    # strides of the final view
    strides = (batch_stride, s0 * st1, s1 * st2, s3) + (s0, s1)

    subs = np.lib.stride_tricks.as_strided(FLOAT_DTYPE(arr), view_shape, strides=strides)
    return subs
