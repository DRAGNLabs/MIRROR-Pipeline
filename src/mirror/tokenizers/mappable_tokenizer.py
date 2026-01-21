from abc import ABC, abstractmethod
from typing import Sequence

import torch

from mirror.types import TokenTensor, TokenBatch


class MappableTokenizer(ABC):
    @property
    @abstractmethod
    def tokenization_id(self) -> str:
        """Unique Identifier for a given tokenizer."""
        pass

    @abstractmethod
    def pad_token_id(self) -> int:
        """The integer ID used for padding."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        By Leaving out torch.tensor we are making the code more saveable, 
        and supposedly HF will wrap this with a C++ / Rust implementation to make it fast.
        The above is hopefully true for encode_batch, and hf_map.
        """
        pass

    def encode_batch(self, texts: Sequence[str]) -> List[List[str]]:
        return [self.encode(text) for text in texts]

    def 

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    def decode_batch(self, token_batch: Sequence[List[int]]) -> Sequence[str]:
        return [self.decode(tokens) for tokens in token_batch]

    @property
    def pad_token_id(self) -> int:
        pass

    def get_hf_map_function(self, text_column: str = "text"):
        """
        Returns a function compatible with dataset.map(..., batched=True).
        
        Usage:
            dataset.map(tokenizer.get_hf_map_function("content_label"), batched=True)
        """
        def hf_wrapper(batch: Dict[str, Any]) -> Dict[str, Any]:
            texts = batch[text_column]
            input_ids = self.encode_batch(texts)
            return {"input_ids": input_ids}
            
        return hf_wrapper

# Removes the hard dependency on torch for the base class strictly for 
# data processing to ensure the saved files are platform-agnostic.

# Depending on your type definitions, TokenTensor might need to be Union[List[int], torch.Tensor]

# We can now load the data using a data loader, 
# so that we'll be able to quickly access batches,
# and convert seamlessly into tensors.
