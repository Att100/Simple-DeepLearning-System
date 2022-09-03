from TinyDL.core.types import Vcpu
from TinyDL.core.autograd.function import Function

from typing import Tuple
import numpy as np


__all__ = ["Exp", "Ln"]

# operators
class Exp(Function):
    name = 'exp'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        return np.exp(x)

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        return np.exp(x) * gradient, None

class Ln(Function):
    name = 'ln'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param: dict=None) -> Vcpu:
        return np.log(x)

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        return (1 / x) * gradient, None