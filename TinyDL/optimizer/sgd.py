from typing import Dict
import numpy as np

from TinyDL.core.tensor import Tensor
from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, 
                    parameters: Dict, 
                    learning_rate: float = 1e-2,
                    momentum: float = 0.9,
                    weight_decay: float = 0) -> None:
        super().__init__(parameters, learning_rate=learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state_dict = {k: {
            'm': np.zeros_like(v.grad),
            'momentum': self.momentum,
            'weight_decay': self.weight_decay} for k, v in self.parameters.items()}

    def step(self):
        for key, val in self.parameters.items():
            momentum = self.state_dict[key]['momentum']
            wd = self.state_dict[key]['weight_decay']
            g = self.parameters[key].grad + self.parameters[key].data * wd
            self.state_dict[key]['m'] = momentum * self.state_dict[key]['m'] + (1 - momentum) * g
            val.data -= self.state_dict[key]['m'] * self.lr
