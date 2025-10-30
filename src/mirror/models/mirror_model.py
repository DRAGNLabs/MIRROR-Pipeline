from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod

from mirror.types import TokenBatch, AttentionMaskBatch, Loss


class MirrorModel(ABC, nn.Module):
    @abstractmethod
    def training_step(
            self,
            tokens: TokenBatch,
            attention_mask: AttentionMaskBatch
    ) -> Loss:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass
