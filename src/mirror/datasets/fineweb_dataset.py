from typing import Literal, cast

import numpy as np
from datasets import Dataset, DatasetDict

from mirror.datasets.dataset_util import load_hf_dataset, slice_by_fraction
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import TextRow

hf_dataset_path = 'HuggingFaceFW/fineweb-edu'
hf_dataset_name = 'sample-10BT'
target_token_count = 2_000_000_000


class FinewebDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
        self,
        head: int | None = None,
        skip: int | None = None,
        split: Literal['train'] = 'train',
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ):
        super().__init__()

        self._ds = cast(DatasetDict, load_hf_dataset(
            hf_dataset_path,
            hf_dataset_name,
            self._process,
        ))[split]

        self._ds = slice_by_fraction(self._ds, start_fraction, end_fraction)

        if skip:
            self._ds = self._ds.select(range(skip, len(self._ds)))

        if head:
            self._ds = self._ds.select(range(head))

    def _process(self, ds: DatasetDict | Dataset) -> DatasetDict:
        assert isinstance(ds, DatasetDict)
        return DatasetDict({split: self._truncate(d) for split, d in ds.items()})

    def _truncate(self, ds: Dataset) -> Dataset:
        cumulative = np.cumsum(np.asarray(ds['token_count'], dtype=np.int64))
        cutoff = int(np.searchsorted(cumulative, target_token_count, side='right')) + 1
        return ds.select(range(cutoff))

    def to_row_type(self, ds_row: dict) -> TextRow:
        return TextRow(text=ds_row['text'])

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
        return self.ds[index]['text']
