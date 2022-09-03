import numpy as np

from TinyDL.core.tensor import Tensor
import TinyDL.core.functional as F
from TinyDL.nn.module import Module

class Linear(Module):
    module_name = "Linear"
    def __init__(self, 
                    in_features: int, 
                    out_features: int, 
                    bias: bool = True, 
                    name: str = "") -> None:
        super().__init__(name=name)
        self.requires_bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            data=np.random.randn(in_features, out_features) * 0.1,
            requires_grad=True,
            name=self.name+"_weight",
            tracer=self.tracer)
        self.bias = None if not self.requires_bias else Tensor(
            data=np.zeros((1, out_features)),
            requires_grad=True,
            name=self.name+"_bias",
            tracer=self.tracer
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.requires_bias:
            return F.matmul(x, self.weight) + self.bias
        else:
            return F.matmul(x, self.weight)

