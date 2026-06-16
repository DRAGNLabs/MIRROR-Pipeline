from typing import Literal, cast

from datasets import DatasetDict
from typed_datasets import TypedDataset

from mirror.datasets.dataset_util import load_hf_dataset, just_text_row
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import TextRow
from mirror.util import _ds_cache_path_context

hf_dataset_path = 'stanfordnlp/imdb'


class ImdbDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> TypedDataset[TextRow]:
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

        raw = cast(DatasetDict, load_hf_dataset(hf_dataset_path))[split]
        ds = TypedDataset[TextRow](raw)

        if skip:
            ds = ds.skip(skip)
        if head:
            ds = ds.take(head)

        with _ds_cache_path_context():
            self._ds = ds.map(just_text_row, remove_columns=list(ds.columns))

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
       return self.ds[index]['text']
