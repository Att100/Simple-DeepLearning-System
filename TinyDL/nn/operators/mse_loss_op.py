import numpy as np
from typing import Tuple

from TinyDL.core.autograd.function import Function
from TinyDL.core.types import Vcpu

class MSELossOp(Function):
    name = 'mse_loss'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        return (x - y)**2

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        return 2 * (x - y) * gradient, None