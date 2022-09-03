from TinyDL.core.types import V, Vcpu, Vcuda
from TinyDL.core.cuda.cuarray import cuArray
import numpy as np

class Variable(object):
    """
    # Variable (Autograd Base Class)

    ## Attribute
            data (V): data that stored in this variable
            requires_grad (bool): need to compute gradient for this variable
            is_leaf (bool): whether this variable is a leaf node
            grad (V): gradient of the data
            l_child (Variable): left children of variable
            r_child (Variable): right children of variable
            father (Variable): father node of this variable 
            operator (Function): the operator that produce this variable
            device (str): cpu or cuda device
            name (str): variable name
            operator_param (dict): the parameter of operators 
            cache (Any): cache of operators

    ## Args
            data (Vcpu): data that will stored in this variable
            requires_grad (bool): need to compute gradient for this variable
            device (bool): cpu or cuda device
            name (str): variable name
    """
    def __init__(self, 
        data: Vcpu, 
        requires_grad: bool = False, 
        device: str = 'cpu',
        name: str = "") -> None:

        super().__init__()
        from .function import Function

        self.data: V = data
        self.requires_grad: bool = requires_grad
        self.is_leaf: bool = False
        self.grad: V = None
        self.l_child: Variable = None
        self.r_child: Variable = None
        self.father: Variable = None
        self.operator: Function = None
        self.device: str = device
        self.name: str = name
        self.operator_param: dict = None
        self.cache = None

        if self.name != "":
            self.is_leaf = True
        
        if device == 'cuda' and not isinstance(self.data, cuArray):
            self.cuda()
        if device == 'cpu' and isinstance(self.data, cuArray):
            self.cpu()

        self._init_node_data()

    def __del__(self):
        self.data = None
        self.grad = None
        self.cache = None
        self.l_child = None
        self.r_child = None
        
    def _constant_like_data(self, fill=0):
        """
        create data with the shape of self.data
        """
        if self.device == 'cpu':
            if isinstance(self.data, np.ndarray):
                out = np.ones_like(self.data) * fill
            else:
                out = fill
        return out

    def _init_node_data(self):
        self.grad = self._constant_like_data()

    def _add_to_graph(self, node, x, y, op, param: dict = None):
        """
        create connection between children node and parent node
        """
        node.l_child = x
        x.father = node
        if y is not None and isinstance(y, Variable):
            y.father = node
            node.r_child = y
        node.operator = op
        node.operator_param = param
        return node

    def _wrap_variable(self, x, y, op, param: dict = None):
        new_variable = Variable(
            op.forward(x.data, y.data if y is not None else None, param),
            requires_grad=self.requires_grad,
            device=x.device
        )
        return self._add_to_graph(new_variable, x, y, op)

    def _wrap_cache_with_param(self):
        """
        wrap operator cache into param dict for backward inputs
        """
        if self.cache is not None:
            if self.operator_param is not None:
                param = self.operator_param.copy()
                param['cache'] = self.cache
                return param
            else:
                return {'cache': self.cache}
        else:
            return self.operator_param
    
    def backward(self, gradient=None, retain_graph=False, is_root=True):
        """
        BP and compute gradient
        """
        self._gradient_backward(gradient, is_root)
        if not retain_graph:
            self._destroy_backward()  # destroy graph except leaf node to release memory
        
    def _gradient_backward(self, gradient=None, is_root=True):
        if is_root and gradient is None:
            gradient = self._constant_like_data(fill=1)
        if not self.requires_grad:
            return
        self.grad = self.grad + gradient
        if not self.is_leaf:
            left_grad, right_grad = self.operator.gradient(
                gradient, 
                self.l_child.data, 
                self.r_child.data if self.r_child is not None else None, 
                self._wrap_cache_with_param())
            if self.l_child is not None:
                if self.l_child.requires_grad:
                    self.l_child._gradient_backward(gradient=left_grad, is_root=False)
            if self.r_child is not None:
                if self.r_child.requires_grad:
                    self.r_child._gradient_backward(gradient=right_grad, is_root=False)

    def _destroy_backward(self):
        if not self.is_leaf:
            if self.l_child is not None:
                if self.l_child.requires_grad:
                    self.l_child._destroy_backward()
            if self.r_child is not None:
                if self.r_child.requires_grad:
                    self.r_child._destroy_backward()
            self.__del__()
        else:
            self.l_child = None
            self.r_child = None
    
    def detach(self):
        pass

    def cpu(self):
        self.device = 'cpu'

    def cuda(self):
        self.device = 'cuda'
    

    

