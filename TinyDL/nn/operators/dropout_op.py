import numpy as np
from typing import Tuple

from TinyDL.core.autograd.function import Function
from TinyDL.core.types import Vcpu

class DropoutOp(Function):
    name = 'dropout'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        rate = param['rate']
        randmask = np.random.uniform(0, 1, size=x.data.shape)
        mask = np.zeros_like(randmask)
        mask[randmask>=rate] = 1 / (1 - rate)
        return x * mask, mask

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        mask = param['cache']
        return mask * gradient, None