from abc import ABC, abstractmethod
from typing import Any, Mapping

from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset


class MirrorFormatter[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT](ABC):
    @abstractmethod
    def format_data(self, data_source: MirrorDataset[RawT]) -> TypedDataset[FormattedT]:
        pass

    @abstractmethod
    def collate(self, examples: list[FormattedT]) -> BatchT:
        pass

