from abc import ABC, abstractmethod
from typing import Sequence, List

import torch

class MirrorPreprocessor[RawT, ProcessedT, BatchT](ABC):
    @property
    @abstractmethod
    def tokenization_id(self) -> str:
        """Unique Identifier for a given tokenizer."""
        pass

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

    def create_hf_map_function(self, data_column: str = 'text', output_column: str = 'input_ids'):
        """
        Returns a function to be used in a ds.map(...) function call.
        """
        def hf_map_callable(input):
            input[output_column] = self.encode(input[data_column])
            return input
        
        return hf_map_callable
