from torch.optim import Optimizer
import torch.nn as nn

from mirror.models.trainable_model import TrainableModel
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.types import Loss


class Intervention[RawT, ProcessedT, BatchT](
    TrainableModel[RawT, ProcessedT, BatchT]
):
    def __init__(self, target: TrainableModel[RawT, ProcessedT, BatchT]):
        super().__init__()
        self.target = target

    @property
    def preprocessor(self) -> MirrorPreprocessor[RawT, ProcessedT, BatchT]:
        return self.target.preprocessor

    def training_step(self, batch: BatchT) -> Loss:
        return self.target.training_step(batch)

    def configure_optimizers(self) -> Optimizer:
        return self.target.configure_optimizers()

    def mlp_modules(self) -> list[nn.Module]:
        return self.target.mlp_modules()
