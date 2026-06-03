from typing import Literal, cast

from datasets import DatasetDict
from typed_datasets import TypedDataset

from mirror.datasets.dataset_util import load_hf_dataset, to_text_row
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import TextRow
from mirror.util import _ds_cache_path_context

hf_dataset_path = 'Salesforce/wikitext'
hf_dataset_name = 'wikitext-2-raw-v1'


class WikitextDataset(MirrorDataset[TextRow]):
    @property
    def ds(self) -> TypedDataset[TextRow]:
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
        super().__init__()

        raw = cast(DatasetDict, load_hf_dataset(hf_dataset_path, hf_dataset_name))[split]
        typed: TypedDataset[TextRow] = TypedDataset(raw)

        if skip:
            typed = typed.skip(skip)
        if head:
            typed = typed.take(head)

        with _ds_cache_path_context():
            typed = typed.filter(lambda row: len(row['text']) > 0)
            self._ds = typed.map(to_text_row, remove_columns=list(typed.columns))

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
       return self.ds[index]['text']
