from ..core.autograd.utils import DyGraphTracer
from ..core.tensor import Tensor
from typing import Any, Dict


class Module(object):
    """
    # Module (Base Class for NN modules)

    ## Attributes

            name (str): the name of this module
            tracer (str): used to track the structure of computation graph
            state_dict (dict): the reference to the weights of all the weights 
            module_name (str): the type of this module

    ## Args
    
            name (str): the name of this module
    """
    module_name = ""
    def __init__(self, name: str = "m") -> None:
        super().__init__()
        self.name = name
        self.tracer = None
        self.state_dict = {}

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return self.forward(*args, **kwds)

    def forward(self, *args: Any, **kwds: Any) -> Tensor:
        """
        this method need to be implemented
        """
        pass

    def _module_to_cuda(self):
        pass

    def _module_to_cpu(self):
        pass

    def set_tracer(self, tracer: DyGraphTracer):
        self.tracer = tracer

    def add_module(self, key, module):
        """
        add sub module from the outside of __init__ 
        """
        self.__setattr__(key, module)

    def init(self, tracer: DyGraphTracer = None):
        """
        initialize the module, this method must be called after the 
        construction of the object of outmost layer (module/container)
        """
        self.set_tracer(tracer)
        self._init_state_dict_keys(self, self.name)
        
    def _init_state_dict_keys(self, module, prev_name):
        """
        set individual name for every weight tensor 
        """
        n_module = 0
        for key, val in module.__dict__.items():
            if isinstance(val, Tensor):
                if key in ['weight', 'bias', 'running_mean', 'running_var']:
                    val.name = prev_name + "_" + key
                    val.tracer = self.tracer
                    if self.tracer is not None:
                        val.tracer.step(val)
                    self.state_dict[val.name] = val
            if isinstance(val, Module):
                n_module += 1
                self._init_state_dict_keys(val, prev_name+"_"+str(n_module))

    def load_state_dict(self, state_dict: Dict):
        pass

    def parameters(self):
        """
        return the state dict of this module 
        """
        return self.state_dict

    def train(self):
        """
        set model to train mode
        """
        def set_train_mode(module):
            for key, val in module.__dict__.items():
                if isinstance(val, Module):
                    if val.module_name == "Dropout":
                        val.train_mode = True
                    elif val.module_name == "BatchNorm1d":
                        val.train_mode = True
                    else:
                        set_train_mode(val)
        set_train_mode(self)

    def eval(self):
        """
        set model to eval mode, (skip dropout, use fixed bn param)
        """
        def set_eval_mode(module):
            for key, val in module.__dict__.items():
                if isinstance(val, Module):
                    if val.module_name == "Dropout":
                        val.train_mode = False
                    elif val.module_name == "BatchNorm1d":
                        val.train_mode = False
                    else:
                        set_eval_mode(val)
        set_eval_mode(self)

                
