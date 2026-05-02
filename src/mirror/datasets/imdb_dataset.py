from typing import Literal, cast

from datasets import Dataset, DatasetDict
from mirror.datasets.dataset_util import load_hf_dataset
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.dict_types import TextRow

hf_dataset_path = 'stanfordnlp/imdb'


class ImdbDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
        self,
        head: int | None = None,
        skip: int | None = None,
        split: Literal['train'] | Literal['test'] | Literal['unsupervised'] = 'train',
    ):
        """
        Args:
            head: how many examples to include. None includes the whole split.
            skip: how many examples to skip from the start.
            split: which dataset split to use. 'unsupervised' is the union of
                'train' and 'test'
        """
        super().__init__()

        self._ds = cast(DatasetDict, load_hf_dataset(hf_dataset_path))[split]
        if skip:
            self._ds = self._ds.select(range(skip, len(self._ds)))
        if head:
            self._ds = self._ds.select(range(head))

    def to_row_type(self, ds_row: dict) -> TextRow:
        return TextRow(text=ds_row['text'])

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
       return self.ds['text'][index]
