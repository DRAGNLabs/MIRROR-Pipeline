from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerBase


class MirrorPreprocessor[RawT, ProcessedT, BatchT](ABC):
    @abstractmethod
    def preprocess_example(self, example: RawT) -> ProcessedT:
        pass

    @abstractmethod
    def collate(self, examples: list[ProcessedT]) -> BatchT:
        pass


class InferenceFriendlyPreprocessor(ABC):
    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase: ...

