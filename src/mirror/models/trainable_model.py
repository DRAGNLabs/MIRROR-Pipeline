from abc import abstractmethod

from torch import nn
from torch.optim import Optimizer

from mirror.preprocessors.has_preprocessor import HasPreprocessor
from mirror.types import Loss


class TrainableModel[RawT, ProcessedT, BatchT](
    HasPreprocessor[RawT, ProcessedT, BatchT],
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
