import numpy as np

from TinyDL.core.tensor import Tensor
import TinyDL.nn.functional as F
from TinyDL.nn.module import Module

class ReLu(Module):
    module_name = "ReLu"
    def __init__(self, 
                    inplace: bool = False,
                    name: str = "") -> None:
        super().__init__(name=name)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)