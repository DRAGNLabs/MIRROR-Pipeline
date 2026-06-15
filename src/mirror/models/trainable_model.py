from abc import abstractmethod
from typing import Any, Mapping

from torch import nn
from torch.optim import Optimizer

from mirror.formatters.has_formatter import HasFormatter
from mirror.types import Loss


class TrainableModel[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT](
    HasFormatter[RawT, FormattedT, BatchT],
    nn.Module,
):
    @abstractmethod
    def training_step(self, batch: BatchT) -> Loss:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass

    def mlp_modules(self) -> list[nn.Module]:
        return []
