from lightning import Fabric
from torch import Tensor
from torch.optim import Optimizer

from mirror.optimization.optimization_strategy import OptimizationStrategy


class DefaultOptimizationStrategy(OptimizationStrategy):
    def step(
            self,
            *,
            fabric: Fabric,
            optimizer: Optimizer,
            loss: Tensor,
            **kwargs,
    ) -> bool:
        optimizer.zero_grad()
        fabric.backward(loss)
        optimizer.step()
        return True
