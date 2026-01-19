from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mirror.types import TokenBatch, AttentionMaskBatch, TrainStepOutput
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer

ModelOutT = TypeVar("ModelOutT")

class MirrorModel(ABC, nn.Module, Generic[ModelOutT]):
    @property
    @abstractmethod
    def tokenizer(self) -> MirrorTokenizer:
        pass

    @abstractmethod
    def training_step(
            self,
            tokens: TokenBatch,
            attention_mask: AttentionMaskBatch
    ) -> TrainStepOutput[ModelOutT]:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass
