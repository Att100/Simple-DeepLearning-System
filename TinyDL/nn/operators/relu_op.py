import numpy as np
from typing import Tuple

from TinyDL.core.autograd.function import Function
from TinyDL.core.types import Vcpu

class ReLuOp(Function):
    name = 'relu'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        if param['inplace']:
            x[x < 0] = 0
            return x
        else:
            return np.where(x<0, 0, x)

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        if param['inplace']:
            return np.where(x>0, 1, 0) * gradient, None
        return np.where(x<0, 0, 1) * gradient, None