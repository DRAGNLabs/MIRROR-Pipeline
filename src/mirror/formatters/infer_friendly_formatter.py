from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerBase


class InferFriendlyFormatter(ABC):
    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        pass
