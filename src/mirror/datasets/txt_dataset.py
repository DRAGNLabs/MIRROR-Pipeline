from pathlib import Path
from typing import Literal, cast

from datasets import Dataset, DatasetDict, load_dataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.row_types import TextRow


class TxtDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
            self,
            file_path: str | Path | dict[str, str | Path],
            head: int | None = None,
            split: Literal['train', 'validation', 'test'] = 'train',
    ):
        """
        Args:
            file_path: path to a .txt file where each line is one example,
                or a dict mapping split names to file paths
                (e.g. {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}).
            head: how many examples to include. None includes the whole split.
            split: which dataset split to use.
        """
        super().__init__()

        if isinstance(file_path, dict):
            data_files = {k: str(v) for k, v in file_path.items()}
        else:
            data_files = {split: str(file_path)}

        ds = cast(DatasetDict, load_dataset("text", data_files=data_files))
        ds = ds.filter(lambda row: len(row["text"]) > 0)
        self._ds = ds[split]
        if head:
            self._ds = self._ds.select(range(head))

    def __getitem__(self, index: int) -> TextRow:
        return cast(TextRow, self.ds[index])

    def __len__(self) -> int:
        return len(self.ds)
