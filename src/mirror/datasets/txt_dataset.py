from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.row_types import TextRow


class TxtDataset(MirrorDataset[TextRow]):
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
        
        self.ds: Dataset = cast(Dataset, load_dataset("text", data_files=str(file_path), split="train"))
        self.ds = self.ds.filter(lambda row: len(row["text"]) > 0)
        if head:
            self.ds = self.ds.select(range(head))

    def __getitem__(self, index: int) -> TextRow:
        return cast(TextRow, self.ds[index])

    def __len__(self) -> int:
        return len(self.ds)
