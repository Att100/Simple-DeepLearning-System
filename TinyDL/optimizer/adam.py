from typing import Dict
import numpy as np

from TinyDL.core.tensor import Tensor
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, 
                    parameters: Dict, 
                    learning_rate: float = 1e-3,
                    beta1: float = 0.9,
                    beta2: float = 0.999,
                    epsilon: float = 1e-8,
                    weight_decay: float = 0) -> None:
        super().__init__(parameters, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.state_dict = {k: {
            'm': np.zeros_like(v.grad),
            'v': np.zeros_like(v.grad),
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'epsilon': self.epsilon} for k, v in self.parameters.items()}
        self.counter = 1

    def step(self):
        for key, val in self.parameters.items():
            beta1 = self.state_dict[key]['beta1']
            beta2 = self.state_dict[key]['beta2']
            wd = self.state_dict[key]['weight_decay']
            epsilon = self.state_dict[key]['epsilon']
            g = self.parameters[key].grad + self.parameters[key].data * wd
            self.state_dict[key]['m'] = beta1 * self.state_dict[key]['m'] + (1 - beta1) * g
            self.state_dict[key]['v'] = beta2 * self.state_dict[key]['v'] + (1 - beta2) * (g**2)
            m_c = self.state_dict[key]['m'] / (1 - beta1**self.counter)
            v_c = self.state_dict[key]['v'] / (1 - beta2**self.counter)
            val.data -= (m_c/(np.sqrt(v_c) + epsilon)) * self.lr
        self.counter += 1