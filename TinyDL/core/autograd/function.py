from typing import Tuple

from TinyDL.core.autograd.variable import Variable


class Function(object):
    """
    # Function (Autograd Base Class)

    ## Note:
           1. forward and gradient methods are need to be implemented 
           2. only support 1 element operationa and 2 elements operation
           3. forward and gradient methods should be static methods
    """
    name = ""
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(self, x: Variable = None, y: Variable = None, param=None) -> Variable:
        return None

    @staticmethod
    def gradient(self, 
                    gradient: Variable = None, 
                    x: Variable = None, 
                    y: Variable = None,
                    param=None) -> Tuple[Variable]:
        return None, None

    