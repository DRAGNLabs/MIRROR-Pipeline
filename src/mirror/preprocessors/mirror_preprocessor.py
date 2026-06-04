from abc import ABC, abstractmethod
from typing import Any, Mapping


class MirrorPreprocessor[RawT: Mapping[str, Any], ProcessedT: Mapping[str, Any], BatchT](ABC):
    @abstractmethod
    def preprocess_example(self, example: RawT) -> ProcessedT:
        pass

    @abstractmethod
    def collate(self, examples: list[ProcessedT]) -> BatchT:
        pass

