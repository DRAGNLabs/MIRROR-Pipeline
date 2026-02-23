from abc import ABC, abstractmethod
from typing import Sequence

import torch

from mirror.types import TokenTensor, TokenBatch


class MirrorPreprocessor[RawT, ProcessedT, BatchT](ABC):
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
    def preprocess_example(self, example: RawT) -> ProcessedT:
        pass

    @abstractmethod
    def collate(self, examples: list[ProcessedT]) -> BatchT:
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass
