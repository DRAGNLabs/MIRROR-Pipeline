from torch.optim import Optimizer
from torch import nn
from abc import ABC, abstractmethod
from mirror.types import TrainStepOutput
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor


class MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT](ABC, nn.Module):
    @property
    @abstractmethod
    def preprocessor(self) -> MirrorPreprocessor[RawT, ProcessedT, BatchT]:
        pass

    @abstractmethod
    def training_step(self, batch: BatchT) -> TrainStepOutput[ModelOutputT]:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass

    def mlp_modules(self) -> list[nn.Module]:
        return []
