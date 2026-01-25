from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod
from mirror.types import AttentionMaskBatch, TrainStepOutput
from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer

class MirrorModel[ProcessedT, ModelOutputT](ABC, nn.Module):
    @property
    @abstractmethod
    def tokenizer(self) -> MirrorTokenizer:
        pass

    @abstractmethod
    def training_step(
            self,
            tokens: ProcessedT,
            attention_mask: AttentionMaskBatch
    ) -> TrainStepOutput[ModelOutputT]:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass
