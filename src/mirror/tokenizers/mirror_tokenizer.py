from abc import ABC, abstractmethod
from typing import Sequence

import torch

from mirror.types import TokenTensor, TokenBatch


class MirrorTokenizer(ABC):
    @property
    @abstractmethod
    def tokenization_id(self) -> str:
        pass

    @abstractmethod
    def encode(self, text: str) -> TokenTensor:
        pass

    def encode_batch(self, texts: Sequence[str]) -> TokenBatch:
        return torch.stack([self.encode(text) for text in texts])

    @abstractmethod
    def decode(self, tokens: TokenTensor) -> str:
        pass

    def decode_batch(self, token_batch: TokenBatch) -> Sequence[str]:
        return [self.decode(tokens) for tokens in token_batch]
