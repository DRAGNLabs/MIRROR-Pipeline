from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class InferFriendlyFormatter(ABC):
    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        pass
