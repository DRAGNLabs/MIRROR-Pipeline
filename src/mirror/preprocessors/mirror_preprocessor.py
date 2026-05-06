from abc import ABC, abstractmethod
from typing import Sequence


class MirrorPreprocessor[RawT, ProcessedT, BatchT](ABC):
    @abstractmethod
    def preprocess_example(self, example: RawT) -> Sequence[ProcessedT]:
        pass

    @abstractmethod
    def collate(self, examples: list[ProcessedT]) -> BatchT:
        pass

