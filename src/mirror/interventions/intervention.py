from typing import Any, Mapping
from torch.optim import Optimizer
import torch.nn as nn

from mirror.models.trainable_model import TrainableModel
from mirror.formatters.mirror_formatter import MirrorFormatter
from mirror.types import Loss


class Intervention[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT](
    TrainableModel[RawT, FormattedT, BatchT]
):
    def __init__(self, target: TrainableModel[RawT, FormattedT, BatchT]):
        super().__init__()
        self.target = target

    @property
    def formatter(self) -> MirrorFormatter[RawT, FormattedT, BatchT]:
        return self.target.formatter

    def training_step(self, batch: BatchT) -> Loss:
        return self.target.training_step(batch)

    def configure_optimizers(self) -> Optimizer:
        return self.target.configure_optimizers()

    def mlp_modules(self) -> list[nn.Module]:
        return self.target.mlp_modules()
