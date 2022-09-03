from TinyDL.core.tensor import Tensor
from TinyDL.core.operators.common import *
from TinyDL.core.operators.math import *

__all__ = [
    "add", "sub", "mul", "div", "matmul", "pow", "t", "permute",
    "mean", "sum", "exp", "ln"]

# common
def add(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, Add)

def sub(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, Sub)

def mul(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, Mul)

def div(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, Div)

def matmul(x: Tensor, y: Tensor):
    return x._wrap_tensor(x, y, Matmul)

def pow(x: Tensor, p: float = 2):
    return x._wrap_tensor(x, None, Pow, {'power': p})

def t(x: Tensor):
    return x._wrap_tensor(x, None, Transpose)

def permute(x: Tensor, *args):
    return x._wrap_tensor(x, None, Permute, {'axis': tuple(args)})

def mean(x: Tensor, axis: int = None):
    return x._wrap_tensor(x, None, Mean, {'axis': axis})

def sum(x: Tensor, axis: int = None):
    return x._wrap_tensor(x, None, Sum, {'axis': axis})


# math
def exp(x: Tensor):
    return x._wrap_tensor(x, None, Exp)

def ln(x: Tensor):
    return x._wrap_tensor(x, None, Ln)
