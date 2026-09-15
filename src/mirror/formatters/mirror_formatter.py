from abc import ABC, abstractmethod
from typing import Any, Mapping

from typed_datasets import TypedDataset

from mirror.datasets.data_source import DataSource


class MirrorFormatter[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT](ABC):
    @abstractmethod
    def format_data(self, data_source: DataSource[RawT]) -> TypedDataset[FormattedT]:
        pass

    @abstractmethod
    def collate(self, examples: list[FormattedT]) -> BatchT:
        pass

