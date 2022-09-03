from TinyDL.core.types import Vcpu
from TinyDL.core.autograd.function import Function

from typing import Tuple
import numpy as np


__all__ = [
    "Add", "Sub", "Neg", "Mul", "Div", "Matmul", 
    "Pow", "Transpose", "Permute", "Mean", "Sum"]

class Add(Function):
    """
    # Addition or Element-wise Subtraction

    ## Note:
        1. add (element-wise) x and y on their last two 
           dimensions if x and y are matrix

        2. only support inputs with same number of dimensions 
           which means broadcast is not supported 

        3. this method support the inputs: matrix*matrix, 
           vector*vector, matrix*scaler, scaler*matrix,
           vector*matrix, matrix*vector, scaler*vector,
           vector*scaler and scaler*scaler         
    """
    name = 'add'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu, param=None) -> Vcpu:
        return x + y

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param=None) -> Tuple[Vcpu, Vcpu]:
        # matrix/vector and matrix/vector product
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # (matrix and matrix) or (vector and vector) product
            if x.shape[-2:] == y.shape[-2:]:
                return gradient, gradient
            # matrix/vector and (1, 1) matrix product
            elif x.shape[-2] == 1 and x.shape[-1] == 1:
                return np.sum(gradient, axis=(-1,-2), keepdims=True), gradient
            elif y.shape[-2] == 1 and y.shape[-1] == 1:
                return gradient, np.sum(gradient, axis=(-1,-2), keepdims=True)
            # matrix and vector product
            elif x.shape[-2] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient, axis=-2, keepdims=True), gradient
            elif x.shape[-1] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient, axis=-1, keepdims=True), gradient
            elif 1 not in x.shape[-2:] and y.shape[-2] == 1:
                return gradient, np.sum(gradient, axis=-2, keepdims=True)
            elif 1 not in x.shape[-2:] and y.shape[-1] == 1:
                return gradient, np.sum(gradient, axis=-1, keepdims=True)
            else:
                raise Exception("Shape of x, y not support")
        # scaler and scaler product
        elif not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient, gradient
        # scaler and matrix product
        elif isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient, None
        elif not isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return None, gradient
        else:
            raise Exception("shape of x, y not support")


class Sub(Function):
    """
    # Subtraction or Element-wise Subtraction

    ## Note:
        1. subtract (element-wise) x and y on their last two 
           dimensions if x and y are matrix

        2. only support inputs with same number of dimensions 
           which means broadcast is not supported 

        3. this method support the inputs: matrix*matrix, 
           vector*vector, matrix*scaler, scaler*matrix,
           vector*matrix, matrix*vector, scaler*vector,
           vector*scaler and scaler*scaler         
    """
    name = 'sub'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu, param=None) -> Vcpu:
        return x - y

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param=None) -> Tuple[Vcpu, Vcpu]:
        # matrix/vector and matrix/vector product
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # (matrix and matrix) or (vector and vector) product
            if x.shape[-2:] == y.shape[-2:]:
                return gradient, -gradient
            # matrix/vector and (1, 1) matrix product
            elif x.shape[-2] == 1 and x.shape[-1] == 1:
                return np.sum(gradient, axis=(-1,-2), keepdims=True), -gradient
            elif y.shape[-2] == 1 and y.shape[-1] == 1:
                return gradient, -np.sum(gradient, axis=(-1,-2), keepdims=True)
            # matrix and vector product
            elif x.shape[-2] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient, axis=-2, keepdims=True), -gradient
            elif x.shape[-1] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient, axis=-1, keepdims=True), -gradient
            elif 1 not in x.shape[-2:] and y.shape[-2] == 1:
                return gradient, -np.sum(gradient, axis=-2, keepdims=True)
            elif 1 not in x.shape[-2:] and y.shape[-1] == 1:
                return gradient, -np.sum(gradient, axis=-1, keepdims=True)
            else:
                raise Exception("shape of x, y not support")
        # scaler and scaler product
        elif not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient, -gradient
        # scaler and matrix product
        elif isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient, None
        elif not isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return None, -gradient
        else:
            raise Exception("shape of x, y not support")


class Neg(Function):
    """
    # Negative

    ## Note:
           1. ex. neg(x) = -x, neg(1) = -1
    """
    name = 'neg'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu = None, param=None) -> Vcpu:
        return -x

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param=None) -> Tuple[Vcpu, Vcpu]:
        return -gradient, None


class Mul(Function):
    """
    # Multiplication or Element-wise Multiplication

    ## Note:
        1. multiply (element-wise) x and y on their last two 
           dimensions if x and y are matrix

        2. only support inputs with same number of dimensions 
           which means broadcast is not supported 

        3. this method support the inputs: matrix*matrix, 
           vector*vector, matrix*scaler, scaler*matrix,
           vector*matrix, matrix*vector, scaler*vector,
           vector*scaler and scaler*scaler         
    """
    name = 'mul'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu, param=None) -> Vcpu:
        return x * y

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param=None) -> Tuple[Vcpu, Vcpu]:
        # matrix/vector and matrix/vector product
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # (matrix and matrix) or (vector and vector) product
            if x.shape[-2:] == y.shape[-2:]:
                return gradient*y, gradient*x
            # matrix/vector and (1, 1) matrix product
            elif x.shape[-2] == 1 and x.shape[-1] == 1:
                return np.sum(gradient*y, axis=(-1,-2), keepdims=True), gradient*x
            elif y.shape[-2] == 1 and y.shape[-1] == 1:
                return gradient*y, np.sum(gradient*x, axis=(-1,-2), keepdims=True)
            # matrix and vector product
            elif x.shape[-2] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient*y, axis=-2, keepdims=True), gradient*x
            elif x.shape[-1] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient*y, axis=-1, keepdims=True), gradient*x
            elif 1 not in x.shape[-2:] and y.shape[-2] == 1:
                return gradient*y, np.sum(gradient*x, axis=-2, keepdims=True)
            elif 1 not in x.shape[-2:] and y.shape[-1] == 1:
                return gradient*y, np.sum(gradient*x, axis=-1, keepdims=True)
            else:
                raise Exception("shape of x, y not support")
        # scaler and scaler product
        elif not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient*y, gradient*x
        # scaler and matrix product
        elif isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient*np.ones_like(x)*y, None
        elif not isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return None, gradient*np.ones_like(y)*x
        else:
            raise Exception("shape of x, y not support")


class Div(Function):
    """
    # Division or Element-wise Division

    ## Note:
        1. divide (element-wise) x by y on their last two 
           dimensions if x and y are matrix

        2. only support inputs with same number of dimensions 
           which means broadcast is not supported 

        3. this method support the inputs: matrix*matrix, 
           vector*vector, matrix*scaler, scaler*matrix,
           vector*matrix, matrix*vector, scaler*vector,
           vector*scaler and scaler*scaler         
    """
    name = 'div'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu, param=None) -> Vcpu:
        return x / y

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param=None) -> Tuple[Vcpu, Vcpu]:
        # matrix/vector and matrix/vector product
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # (matrix and matrix) or (vector and vector) product
            if x.shape[-2:] == y.shape[-2:]:
                return gradient*(1/y), -gradient*x*(1/y**2)
            # matrix/vector and (1, 1) matrix product
            elif x.shape[-2] == 1 and x.shape[-1] == 1:
                return np.sum(gradient*(1/y), axis=(-1,-2), keepdims=True), -gradient*x*(1/y**2)
            elif y.shape[-2] == 1 and y.shape[-1] == 1:
                return gradient*(1/y), np.sum(-gradient*x*(1/y**2), axis=(-1,-2), keepdims=True)
            # matrix and vector product
            elif x.shape[-2] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient*(1/y), axis=-2, keepdims=True), -gradient*x*(1/y**2)
            elif x.shape[-1] == 1 and 1 not in y.shape[-2:]:
                return np.sum(gradient*(1/y), axis=-1, keepdims=True), -gradient*x*(1/y**2)
            elif 1 not in x.shape[-2:] and y.shape[-2] == 1:
                return gradient*(1/y), np.sum(-gradient*x*(1/y**2), axis=-2, keepdims=True)
            elif 1 not in x.shape[-2:] and y.shape[-1] == 1:
                return gradient*(1/y), np.sum(-gradient*x*(1/y**2), axis=-1, keepdims=True)
            else:
                raise Exception("shape of x, y not support")
        # scaler and scaler product
        elif not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient*(1/y), -gradient*x*(1/y**2)
        # scaler and matrix product
        elif isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
            return gradient*(1/y), None
        elif not isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return None, -gradient*x*(1/y**2)
        else:
            raise Exception("shape of x, y not support")


class Matmul(Function):
    """
    # Matrix Multiplication

    ## Note:
        1. multiply x and y on their last two dimensions if 
           x and y are matrix

        2. only support inputs with same number of dimensions 
           which means broadcast is not supported 

        3. shape of inputs should satisfy (..., h, w) and (..., w, h)

        4. if number of dimensions is greater than 2, np.matmul will
           be used to do the multiplication
    """
    name = 'matmul'
    @staticmethod
    def forward(x: np.ndarray, y: np.ndarray, param=None) -> np.ndarray:
        if len(x.shape) >= 3:
            return np.matmul(x, y)
        return np.dot(x, y)
        
    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param=None) -> Tuple[Vcpu, Vcpu]:
        lens_shape = len(x.shape)
        if lens_shape >= 3:
            new_axis = [i for i in range(lens_shape-2)] + [lens_shape-1, lens_shape-2]
            return np.matmul(gradient, y.transpose(tuple(new_axis))), \
                np.matmul(x.transpose(tuple(new_axis)), gradient)
        return np.dot(gradient, y.T), np.dot(x.T, gradient)


class Pow(Function):
    """
    # Power

    ## Note:
           1. compute the power of matrix, vector or scaler

           2. ex. out = pow(x, 2), 4 = pow(2, 2)
    """
    name = 'pow'
    @staticmethod
    def forward(x: Vcpu, y: Vcpu, param: dict=None) -> Vcpu:
        assert param != None, "power of pow operator shouldn't be None"
        return x**param['power']

    @staticmethod
    def gradient(gradient :Vcpu, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        assert param != None, "power of pow operator shouldn't be None"
        return param['power']*(x**(param['power']-1))*gradient, None


class Transpose(Function):
    """
    # Matrix Transpose

    ## Note:
           1. only support the input with shape (m, n)
    """
    name = 'transpose'
    @staticmethod
    def forward(x: np.ndarray, y: Vcpu, param: dict=None) -> np.ndarray:
        return x.T

    @staticmethod
    def gradient(gradient :np.ndarray, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        return gradient.T, None


class Permute(Function):
    """
    # Matrix Axis Swapping

    ## Note:
           1. ex. shape of x is (2, 3, 4, 5), shape of x.permute(0, 1, 3, 2)
              is (2, 3, 5, 4)
    """
    name = 'permute'
    @staticmethod
    def forward(x: np.ndarray, y: Vcpu, param: dict=None) -> np.ndarray:
        return x.transpose(param['axis'])

    @staticmethod
    def gradient(gradient :np.ndarray, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        return gradient.transpose(param['axis']), None


class Mean(Function):
    """
    # Matrix Mean
    """
    name = 'mean'
    @staticmethod
    def forward(x: np.ndarray, y: Vcpu, param: dict=None) -> np.ndarray:
        axis = param['axis']
        if axis is not None:
            return np.mean(x, axis=axis, keepdims=True)
        return np.mean(x)

    @staticmethod
    def gradient(gradient :np.ndarray, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        axis = param['axis']
        mask = np.ones_like(x)
        scale = 1
        if axis is not None:
            scale *= x.shape[axis]
        else:
            for dim in x.shape: scale *= dim 
        return mask * (1/scale) * gradient, None


class Sum(Function):
    """
    # Matrix Sum
    """
    name = 'sum'
    @staticmethod
    def forward(x: np.ndarray, y: Vcpu, param: dict=None) -> np.ndarray:
        axis = param['axis']
        if axis is not None:
            return np.sum(x, axis=axis, keepdims=True)
        return np.sum(x)

    @staticmethod
    def gradient(gradient :np.ndarray, x: Vcpu = None, y: Vcpu = None, param: dict=None) -> Tuple[Vcpu, Vcpu]:
        mask = np.ones_like(x)
        return mask * gradient, None