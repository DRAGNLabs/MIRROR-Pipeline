from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import TextRow
from mirror.util import _ds_cache_path_context


class TxtDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> TypedDataset[TextRow]:
        return self._ds

    def __init__(
            self,
            file_path: str | Path,
            head: int | None = None
    ):
        """
        Args:
            file_path: path to a .txt file where each line is one example.
            head: how many examples to include. None includes the whole file.
        """
        super().__init__()

        raw = cast(Dataset, load_dataset("text", data_files=str(file_path), split="train"))
        ds = TypedDataset[TextRow](raw)

        with _ds_cache_path_context():
            ds = ds.filter(lambda row: len(row['text']) > 0)
            if head:
                ds = ds.take(head)
            self._ds = ds

    def __len__(self) -> int:
        return len(self.ds)
