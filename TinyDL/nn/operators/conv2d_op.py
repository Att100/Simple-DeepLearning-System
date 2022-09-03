import numpy as np
from typing import List, Tuple

from TinyDL.core.autograd.function import Function
from TinyDL.core.types import Vcpu

def _conv_out_shape(in_size: int, padding: int, stride: int, ksize: int):
    return (in_size + 2 * padding - ksize) // stride + 1
    
def img2col(inputs: np.ndarray, k_size: list, stride: list):
    """
    img2col (Low Efficency Implementation)

    args:
        inputs: np.ndarray (b, c, h, w)
        k_size, stride: [int, int], [int, int]

    return:
        cols: np.ndarray (b*out_h*out_w, c*k_h*k_w)
    """
    b, c, h, w = inputs.shape
    k_h, k_w, s_h, s_w = k_size[0], k_size[1], stride[0], stride[1]
    out_h = _conv_out_shape(h, 0, s_h, k_h)
    out_w = _conv_out_shape(w, 0, s_w, k_w)
    cols = []
    for i in range(b):
        for j in range(0, h-k_h+1, s_h):
            for k in range(0, h-k_w+1, s_w):
                for c_ in range(c):
                    cols.append(inputs[i, c_, j:j+k_h, k:k+k_h].reshape(k_h*k_w))
    cols = np.array(cols).reshape(out_w*out_h*b, k_h*k_w*c)
    return cols, out_h, out_w
    
def col2img(cols: np.ndarray, k_size: list, stride: list, in_size: list):
    """
    col2img (Low Efficency Implementation)

    args:
        cols: np.ndarray (b*out_h*out_w, c*k_h*k_w)
        k_size, stride: [int, int], [int, int]

    return:
        outputs: np.ndarray (b, c, h, w)
    """
    b, c, h, w = in_size
    k_h, k_w, s_h, s_w = k_size[0], k_size[1], stride[0], stride[1]
    out_h = _conv_out_shape(h, 0, s_h, k_h)
    out_w = _conv_out_shape(w, 0, s_w, k_w)
    _cols = cols.reshape(b, out_h, out_w, c, k_h*k_w)
    outputs = np.zeros((b, c, h, w))
    for i in range(b):
        for j in range(0, h-k_h+1, s_h):
            for k in range(0, h-k_w+1, s_w):
                for c_ in range(c):
                    outputs[i, c_, j:j+k_h, k:k+k_w] += \
                        _cols[i, j, k, c_, :].reshape(k_h, k_w)
    

class Conv2dMulWeightOp(Function):
    name = 'conv2d_mul_weight'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        p_h, p_w = param['padding']
        stride = param['stride']
        k_size = param['k_size']
        in_ch = param['in_channels']
        out_ch = param['out_channels']
        b, c, w, h = x.shape
        if p_h > 0 or p_w > 0:
            pad_x = np.pad(x, pad_width=((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))
        else:
            pad_x = x
        cols_x, out_h, out_w = img2col(pad_x, k_size, stride)  # (b*out_w*out_h, in_ch*k_h*k_w)
        flat_w = y.reshape(out_ch, -1).T  # (k_h*k_w*in_ch, out_ch)
        out = np.dot(cols_x, flat_w)  # z = X * W
        return out.reshape(b, out_h, out_w, out_ch).transpose((0, 3, 1, 2)), cols_x

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        cols_x = param['cache']
        p_h, p_w = param['padding']
        stride = param['stride']
        k_size = param['k_size']
        in_ch = param['in_channels']
        out_ch = param['out_channels']
        gradient_w = gradient.transpose((1, 0, 2, 3)).reshape(out_ch, -1).dot(cols_x)
        gradient_cols_x = gradient.transpose((0, 2, 3, 1)).reshape(-1, out_ch).dot(
            y.reshape(out_ch, -1))
        gradient_x = col2img(gradient_cols_x, k_size, stride, x.shape)
        # unpad
        if p_h > 0 or p_w > 0:
            gradient_x = gradient_x[:, :, p_h:x.shape[2]+p_h, p_w:x.shape[3]+p_w]
        return gradient_x, gradient_w

class Conv2dAddBiasOp(Function):
    name = 'conv2d_add_bias'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        return x + y

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        mask = param['mask']
        return mask * gradient, None