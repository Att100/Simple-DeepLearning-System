from typing import Any
import numpy as np

from TinyDL.core.tensor import Tensor
import TinyDL.core.functional as F
from TinyDL.nn.module import Module

class Conv2d(Module):
    module_name = "Conv2d"
    def __init__(self, 
                    in_channels: int, 
                    out_channels: int, 
                    kernel_size: Any,
                    stride: Any,
                    padding: Any,
                    bias: bool = True, 
                    name: str = "") -> None:
        super().__init__(name=name)
        self.requires_bias = bias
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = Tensor(
            data=np.random.randn(self.out_ch, self.in_ch, self.k_size[0], self.k_size[1]) * 0.1,
            requires_grad=True,
            name=self.name+"_weight",
            tracer=self.tracer)
        self.bias = None if not self.requires_bias else Tensor(
            data=np.zeros((1, self.out_ch, 1, 1)),
            requires_grad=True,
            name=self.name+"_bias",
            tracer=self.tracer
        )
        
    def forward(self, x: Tensor) -> Tensor:
        if self.requires_bias:
            return F.matmul(x, self.weight) + self.bias
        else:
            return F.matmul(x, self.weight)