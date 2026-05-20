from lightning import Fabric
from torch import Tensor
from torch.optim import Optimizer

from mirror.optimization.optimization_strategy import OptimizationStrategy


class GradientAccumulationStrategy(OptimizationStrategy):
    def __init__(self, accumulation_steps: int) -> None:
        self.accumulation_steps = accumulation_steps
        self._batches_accumulated = 0

    def step(
            self,
            *,
            fabric: Fabric,
            optimizer: Optimizer,
            loss: Tensor,
            **kwargs,
    ) -> bool:
        if self._batches_accumulated == 0:
            optimizer.zero_grad()
        fabric.backward(loss / self.accumulation_steps)
        self._batches_accumulated += 1
        if self._batches_accumulated >= self.accumulation_steps:
            optimizer.step()
            self._batches_accumulated = 0
            return True
        return False
