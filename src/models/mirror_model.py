from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
from abc import ABC, abstractmethod
from jaxtyping import Int, Float

from tokenizers.mirror_tokenizer import MirrorTokenizer, TokenBatch

AttentionMaskBatch = Int[Tensor, "b t"]
Loss = Float[Tensor, ""]


class MirrorModel(ABC, nn.Module):
    @property
    @abstractmethod
    def tokenizer(self) -> MirrorTokenizer:
        pass

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
