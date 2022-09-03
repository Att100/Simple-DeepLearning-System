import numpy as np

from TinyDL.core.tensor import Tensor
import TinyDL.nn.functional as F
from TinyDL.nn.module import Module


class Dropout(Module):
    module_name = "Dropout"
    def __init__(self, rate: float = 0.5, name: str = "") -> None:
        super().__init__(name=name)
        self.rate: float = rate
        self.train_mode = True

    def forward(self, x: Tensor) -> Tensor:
        if self.train_mode:
            return F.dropout(x, self.rate)
        else:
            return F.dropout(x, 0)