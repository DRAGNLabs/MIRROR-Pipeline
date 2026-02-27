from abc import ABC, abstractmethod
from typing import Sequence, List, Callable

import torch

class MirrorPreprocessor[RawT, ProcessedT, BatchT](ABC):
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        By Leaving out torch.tensor we are making the code more saveable, 
        and supposedly HF will wrap this with a C++ / Rust implementation to make it fast.
        The above is hopefully true for encode_batch, and hf_map.
        """
        pass

    def encode_batch(self, texts: Sequence[str]) -> Sequence[ProcessedT]:
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
