from abc import ABC, abstractmethod
from typing import Any, Mapping

from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset


class MirrorPreprocessor[RawT: Mapping[str, Any], ProcessedT: Mapping[str, Any], BatchT](ABC):
    @abstractmethod
    def format_data(self, data_source: MirrorDataset[RawT]) -> TypedDataset[ProcessedT]:
        pass

    @abstractmethod
    def collate(self, examples: list[ProcessedT]) -> BatchT:
        pass

