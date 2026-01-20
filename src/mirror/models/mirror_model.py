from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mirror.types import TokenBatch, AttentionMaskBatch, Loss, RawT, ProcessedT, BatchT
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer


class MirrorModel(ABC, nn.Module, Generic[RawT, ProcessedT, BatchT]):
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
