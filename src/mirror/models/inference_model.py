from abc import abstractmethod

from torch import nn

from mirror.preprocessors.has_preprocessor import HasPreprocessor


class InferenceModel[RawT, ProcessedT, BatchT, ModelOutputT](
    HasPreprocessor[RawT, ProcessedT, BatchT],
    nn.Module,
):
    @abstractmethod
    def forward(self, batch: BatchT) -> ModelOutputT:
        pass
