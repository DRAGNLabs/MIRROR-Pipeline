from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from typed_datasets import TypedDataset, load_typed

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
            head: how many non-empty examples to include. None includes the whole file.
        """
        super().__init__()

        if head is not None:
            stream = load_typed(
                "text", row_type=TextRow, split="train",
                streaming=True, data_files=str(file_path),
            )
            rows = list(stream.filter(lambda row: len(row['text']) > 0).take(head))
            self._ds = TypedDataset.from_list(rows)
        else:
            raw = cast(Dataset, load_dataset("text", data_files=str(file_path), split="train"))
            with _ds_cache_path_context():
                self._ds = TypedDataset(raw).filter(lambda row: len(row['text']) > 0)

    def __len__(self) -> int:
        return len(self.ds)
