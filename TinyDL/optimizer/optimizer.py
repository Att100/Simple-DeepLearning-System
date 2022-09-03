from typing import Dict

from TinyDL.core.tensor import Tensor

class Optimizer(object):
    """
    # Optimizer (Base Class for Optimizer)

    ## Attributes
            parameters (dict): the reference to all the learnabel weight
                                tensor of model
            lr (float): learning rate

    ## Args
            parameters (dict): the reference to all the learnabel weight
                                tensor of model
            lr (float): learning rate
    """
    def __init__(self, 
                    parameters: Dict, 
                    learning_rate: float = 1e-2) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        """
        update weights
        """
        for key, val in self.parameters.items():
            val.data -= val.grad * self.lr

    def zero_grad(self):
        """
        clear gradient of all the learnabel weight
        """
        for key, val in self.parameters.items():
            val.grad *= 0