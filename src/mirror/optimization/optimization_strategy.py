from abc import ABC, abstractmethod

from lightning import Fabric
from torch import Tensor
from torch.optim import Optimizer


class OptimizationStrategy(ABC):
    @abstractmethod
    def step(
            self,
            *,
            fabric: Fabric,
            optimizer: Optimizer,
            loss: Tensor,
            batch_idx: int,
    ) -> bool:
        """Run the backward pass and (conditionally) the optimizer step.

        Returns True when the optimizer was actually stepped.
        """
