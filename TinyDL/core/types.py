from typing import TypeVar
import numpy as np

from TinyDL.core.cuda.cuarray import cuArray

# typing
V = TypeVar("V", np.ndarray, cuArray, float, int)
Vcpu = TypeVar("Vcpu", np.ndarray, float, int)
Vcuda = TypeVar("Vcuda", cuArray, float, int)