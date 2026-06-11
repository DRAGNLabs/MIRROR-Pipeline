from typing import Any, Mapping
from torch.optim import Optimizer
from torch import nn
from abc import ABC, abstractmethod
from mirror.types import TrainStepOutput
from mirror.formatters.mirror_formatter import MirrorFormatter


class MirrorModel[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT, ModelOutputT](ABC, nn.Module):
    @property
    @abstractmethod
    def formatter(self) -> MirrorFormatter[RawT, FormattedT, BatchT]:
        pass

    @abstractmethod
    def training_step(self, batch: BatchT) -> TrainStepOutput[ModelOutputT]:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass

    def mlp_modules(self) -> list[nn.Module]:
        return []
