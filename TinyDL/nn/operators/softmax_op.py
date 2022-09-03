import numpy as np
from typing import Tuple

from TinyDL.core.autograd.function import Function
from TinyDL.core.types import Vcpu

class SoftmaxOp(Function):
    name = 'softmax'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        tmp = np.max(x, axis=1, keepdims=True)
        x = np.exp(x-tmp)
        tmp = np.sum(x, axis=1, keepdims=True)
        out = x / tmp
        return out

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        return None, None