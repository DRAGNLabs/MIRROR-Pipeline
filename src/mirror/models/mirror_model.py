from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Generic

from mirror.types import AttentionMaskBatch, ProcessedT, TokenBatch, TrainStepOutput
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer

class MirrorModel(ABC, nn.Module, Generic[ProcessedT]):
    @property
    @abstractmethod
    def tokenizer(self) -> MirrorTokenizer:
        pass

    @abstractmethod
    def training_step(
            self,
            tokens: TokenBatch,
            attention_mask: AttentionMaskBatch
    ) -> TrainStepOutput[ProcessedT]:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass
