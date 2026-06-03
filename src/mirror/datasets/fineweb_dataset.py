from typing import Literal, cast

import numpy as np
from datasets import DatasetDict
from typed_datasets import TypedDataset

from mirror.datasets.dataset_util import load_hf_dataset, just_text_row
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import TextRow
from mirror.util import _ds_cache_path_context

hf_dataset_path = 'HuggingFaceFW/fineweb-edu'
hf_dataset_name = 'sample-10BT'
target_token_count = 2_000_000_000


class FinewebRow(TextRow):
    id: str
    dump: str
    url: str
    file_path: str
    language: str
    language_score: float
    token_count: int
    score: float
    int_score: int


class FinewebDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> TypedDataset[TextRow]:
        return self._ds

    def __init__(
        self,
        head: int | None = None,
        skip: int | None = None,
        split: Literal['train'] = 'train',
    ):
        super().__init__()

        raw = cast(DatasetDict, load_hf_dataset(hf_dataset_path, hf_dataset_name))[split]
        ds = self._truncate(TypedDataset[FinewebRow](raw))

        if skip:
            ds = ds.skip(skip)
        if head:
            ds = ds.take(head)

        with _ds_cache_path_context():
            self._ds = ds.map(just_text_row, remove_columns=list(ds.columns))

    def _truncate(self, ds: TypedDataset[FinewebRow]) -> TypedDataset[FinewebRow]:
        cumulative = np.cumsum(np.asarray(ds.unwrap()['token_count'], dtype=np.int64))
        cutoff = int(np.searchsorted(cumulative, target_token_count, side='right')) + 1
        return ds.take(cutoff)

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
        return self.ds[index]['text']
