from abc import ABC, abstractmethod
from typing import Any, Mapping

from typed_datasets import TypedDataset


class DataSource[RowT: Mapping[str, Any]](ABC):
    @property
    @abstractmethod
    def ds(self) -> TypedDataset[RowT]:
        pass
