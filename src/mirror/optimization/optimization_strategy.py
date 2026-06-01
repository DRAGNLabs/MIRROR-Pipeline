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

    def expected_optimization_steps(self, total_batches: int) -> int:
        """How many times step() will return True over `total_batches` batches.

        Used to size the LR scheduler. Override in strategies that step less often
        than every batch.
        """
        return total_batches
