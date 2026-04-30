from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.dict_types import TextRow


class TxtDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
            self,
            file_path: str | Path,
            head: int | None = None
    ):
        """
        Args:
            file_path: path to a .txt file where each line is one example.
            head: how many examples to include. None includes the whole split.
        """
        super().__init__()

        self._ds = cast(Dataset, load_dataset("text", data_files=str(file_path), split="train"))
        self._ds = self._ds.filter(lambda row: len(row["text"]) > 0)
        if head:
            self._ds = self._ds.select(range(head))

    def to_row_type(self, ds_row: dict) -> TextRow:
        return TextRow(text=ds_row['text'])

    def __len__(self) -> int:
        return len(self.ds)
