from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from mirror.datasets.dataset_util import slice_by_fraction, take
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import TextRow


class TxtDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
            self,
            file_path: str | Path,
            head: int | None = None,
            start_fraction: float = 0.0,
            end_fraction: float = 1.0,
    ):
        """
        Args:
            file_path: path to a .txt file where each line is one example.
            head: how many examples to include. None includes the whole split.
            start_fraction, end_fraction: take rows[int(start*n):int(end*n)]. Applied before head.
        """
        super().__init__()

        self._ds = cast(Dataset, load_dataset("text", data_files=str(file_path), split="train"))
        self._ds = self._ds.filter(lambda row: len(row["text"]) > 0)
        self._ds = slice_by_fraction(self._ds, start_fraction, end_fraction)
        self._ds = take(self._ds, head=head)

    def to_row_type(self, ds_row: dict) -> TextRow:
        return TextRow(text=ds_row['text'])

    def __len__(self) -> int:
        return len(self.ds)
