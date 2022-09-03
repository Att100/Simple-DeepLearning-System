import numpy as np
from typing import Tuple

import TinyDL
from TinyDL.core.tensor import Tensor
import TinyDL.core.functional as F
from TinyDL.nn.module import Module


class BatchNorm1d(Module):
    module_name = "BatchNorm1d"
    def __init__(self, 
                    num_features: int, 
                    eps: float = 1e-5, 
                    momentum: float = 0.1, 
                    name: str = "") -> None:
        super().__init__(name=name)
        self.num_features = num_features
        self.eps = eps
        self.running_decay = 1 - momentum

        self.mean = None
        self.var = None
        self.weight = Tensor(
            data=np.ones((1, num_features)), requires_grad=True, name=self.name+"_weight")
        self.bias = Tensor(
            data=np.zeros((1, num_features)), requires_grad=True, name=self.name+"_bias")
        self.running_mean = Tensor(data=np.zeros((1, num_features)), name=self.name+"_running_mean")
        self.running_var = Tensor(data=np.ones((1, num_features)), name=self.name+"_running_var")
        self.train_mode = True

    def forward(self, x: Tensor) -> Tensor:
        if self.train_mode:
            self.mean = TinyDL.mean(x, axis=0)  
            self.var = TinyDL.mean(TinyDL.pow(x - self.mean), axis=0)
            self.running_mean.data = self.running_decay * self.running_mean.data + \
                (1 - self.running_decay) * self.mean.data
            self.running_var.data = self.running_decay * self.running_var.data + \
                (1 - self.running_decay) * self.var.data
            x_hat = (x - self.mean) / TinyDL.pow(self.var+self.eps, p=0.5)  # normalize
        else:
            x_hat = (x - self.running_mean) / TinyDL.pow(self.running_var+self.eps, p=0.5)  # normalize
        out = x_hat * self.weight + self.bias  # scale and shift
        return out
