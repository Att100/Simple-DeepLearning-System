import numpy as np
from typing import Tuple

from TinyDL.core.autograd.function import Function
from TinyDL.core.types import Vcpu
from .softmax_op import SoftmaxOp

class CrossEntropyLossOp(Function):
    name = 'cross_entropy_loss'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        logits = SoftmaxOp.forward(x, None)
        loss = (-1) * y * np.log(logits)
        return np.sum(loss, axis=1, keepdims=True)
            
    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        return (SoftmaxOp.forward(x, None) - y) * gradient, None