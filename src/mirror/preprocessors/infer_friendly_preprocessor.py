from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerBase


class InferFriendlyPreprocessor(ABC):
    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        pass
