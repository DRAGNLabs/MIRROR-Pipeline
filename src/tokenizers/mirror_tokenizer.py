from abc import ABC, abstractmethod
from typing import Sequence

from jaxtyping import Int
import torch
from torch import Tensor

from datasets.mirror_dataset import TokenTensor

TokenBatch = Int[Tensor, "b t"]


class MirrorTokenizer(ABC):
    @property
    @abstractmethod
    def tokenization_id(self) -> str:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> TokenTensor:
        pass

    def tokenize_batch(self, texts: Sequence[str]) -> TokenBatch:
        return torch.stack([self.tokenize(text) for text in texts])
