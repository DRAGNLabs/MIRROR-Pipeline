from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod
from mirror.types import Loss
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor


class MirrorModel[RawT, ProcessedT, BatchT](ABC, nn.Module):
    @property
    @abstractmethod
    def preprocessor(self) -> MirrorPreprocessor:
        pass

    @abstractmethod
    def training_step(self, batch: BatchT) -> Loss:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass
