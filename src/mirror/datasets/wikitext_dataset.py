from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.row_types import TextRow

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

hf_dataset_path = 'Salesforce/wikitext'
hf_dataset_name = 'wikitext-2-raw-v1'


class WikitextDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
        self,
        head: int | None = None,
        skip: int | None = None,
        split: Literal['train'] | Literal['validation'] | Literal['test'] = 'train',
    ):
        """
        Args:
            head: how many examples to include. None includes the whole split.
            skip: how many examples to skip from the start.
            split: which dataset split to use.
        """
        from datasets import DatasetDict
        from mirror.datasets.dataset_util import load_hf_dataset

        super().__init__()

        self._ds = cast(DatasetDict, load_hf_dataset(
            hf_dataset_path,
            hf_dataset_name,
            self._process,
        ))[split]

        if skip:
            self._ds = self._ds.select(range(skip, len(self._ds)))

        if head:
            self._ds = self._ds.select(range(head))

    def _process(self, ds: DatasetDict | Dataset) -> DatasetDict | Dataset:
        return ds.filter(lambda row: len(row['text']) > 0)

    def __getitem__(self, index: int) -> TextRow:
        return cast(TextRow, self.ds[index])

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
       return self.ds[index]['text']
