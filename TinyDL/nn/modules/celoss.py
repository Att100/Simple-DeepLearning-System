import TinyDL
from TinyDL.core.tensor import Tensor
import TinyDL.nn.functional as F
from TinyDL.nn.module import Module

class CrossEntropyLoss(Module):
    module_name = "CrossEntropyLoss"
    def __init__(self, 
                    reduction: bool = 'mean',
                    name: str = "") -> None:
        super().__init__(name=name)
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        out = F.cross_entropy_loss(pred, target)
        if self.reduction == "mean":
            return TinyDL.mean(out)
        elif self.reduction == "sum":
            return TinyDL.sum(out)
        else:
            return out