import TinyDL
from TinyDL.core.tensor import Tensor
from TinyDL.nn.module import Module
import TinyDL.nn.functional as F

class MSELoss(Module):
    module_name = "MSELoss"
    def __init__(self, 
                    reduction: bool = 'mean',
                    name: str = "") -> None:
        super().__init__(name=name)
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        out = F.mse_loss(pred, target)
        if self.reduction == "mean":
            return TinyDL.mean(out)
        elif self.reduction == "sum":
            return TinyDL.sum(out)
        else:
            return out