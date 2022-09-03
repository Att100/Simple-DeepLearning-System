import numpy as np

from TinyDL.core.tensor import Tensor
from .operators.relu_op import ReLuOp
from .operators.softmax_op import SoftmaxOp
from .operators.cross_entropy_loss_op import CrossEntropyLossOp
from .operators.mse_loss_op import MSELossOp
from .operators.dropout_op import DropoutOp
from .operators.conv2d_op import Conv2dMulWeightOp, Conv2dAddBiasOp


def relu(x: Tensor, inplace: bool = False):
    param = {'inplace': inplace}
    return x._wrap_tensor(x, None, ReLuOp, param)

def dropout(x: Tensor, rate: int = 0.5):
    """
    No Evaluation Mode!!
    """
    param = {'rate': rate}
    return x._wrap_tensor(x, None, DropoutOp, param)

def softmax(x: Tensor):
    return x._wrap_tensor(x, None, SoftmaxOp)

def cross_entropy_loss(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, CrossEntropyLossOp)

def mse_loss(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, MSELossOp)

def conv2d_mul_weight(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, Conv2dMulWeightOp)

def conv2d_add_bias(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, Conv2dAddBiasOp)
