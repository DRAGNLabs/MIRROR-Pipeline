from abc import ABC, abstractmethod


class MirrorPreprocessor[RawT, ProcessedT, BatchT](ABC):
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
