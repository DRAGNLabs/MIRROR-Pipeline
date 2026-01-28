from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod
from mirror.types import AttentionMaskBatch, TrainStepOutput
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer


class MirrorModel[RawT, ProcessedT, BatchT](ABC, nn.Module):
    @property
    @abstractmethod
    def tokenizer(self) -> MirrorTokenizer:
        pass

    @abstractmethod
    def training_step(self, batch: BatchT) -> Loss:
        pass

    @abstractmethod
    def preprocess_example(self, example: RawT) -> ProcessedT:
        pass

    @abstractmethod
    def collate(self, examples: list[ProcessedT]) -> BatchT:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass
