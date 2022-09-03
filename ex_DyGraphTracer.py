import numpy as np

import TinyDL
import TinyDL.nn as nn
import TinyDL.nn.functional as F
from TinyDL import Tensor
from TinyDL.core.autograd.utils import DyGraphTracer


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(0.2)
        self.linear3 = nn.Linear(16, 10)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.linear1(x)))
        out = F.relu(self.bn2(self.linear2(out)))
        out = self.drop(out)
        out = self.linear3(out)
        return out


if __name__ == "__main__":

    tracer = DyGraphTracer()
    model = Net()
    model.init(tracer)
    x = Tensor(data=np.random.rand(64, 784),
                requires_grad=True,
                name = "x",
                tracer=tracer)
    out = model(x)
    tracer.save_prototxt('./ex_DygraphTracer.prototxt')
            
