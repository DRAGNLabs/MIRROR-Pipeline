from torch.optim import Optimizer
import torch.nn as nn

from mirror.models.mirror_model import MirrorModel
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.types import TrainStepOutput


class Intervention[RawT, ProcessedT, BatchT, ModelOutputT](
    MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def __init__(self, target: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT]):
        super().__init__()
        self.target = target

    @property
    def preprocessor(self) -> MirrorPreprocessor[RawT, ProcessedT, BatchT]:
        return self.target.preprocessor

    def training_step(self, batch: BatchT) -> TrainStepOutput[ModelOutputT]:
        return self.target.training_step(batch)

    def configure_optimizers(self) -> Optimizer:
        return self.target.configure_optimizers()

    def mlp_modules(self) -> list[nn.Module]:
        return self.target.mlp_modules()
