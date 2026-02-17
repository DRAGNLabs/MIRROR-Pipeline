from typing import Literal, cast

from datasets import Dataset, DatasetDict

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.util import load_hf_from_cache_or_download
from mirror.row_types import TextRow

hf_dataset_path = 'Salesforce/wikitext'
hf_dataset_name = 'wikitext-2-raw-v1'


class WikitextDataset(MirrorDataset[TextRow]):
    def __init__(
        self,
        head: int | None = None,
        split: Literal['train'] | Literal['validation'] | Literal['test'] = 'train',
    ):
        """
        Args:
            head: how many examples to include. None includes the whole split.
            split: which dataset split to use.
        """
        super().__init__()
        self.ds: Dataset = cast(DatasetDict, load_hf_from_cache_or_download(
            hf_dataset_path,
            hf_dataset_name,
            self._process,
        ))[split]

        if head: 
            self.ds = self.ds.select(range(head))

    @property
    def dataset_id(self) -> str:
        return f'{hf_dataset_path}/{hf_dataset_name}'

    def _process(self, ds: DatasetDict | Dataset) -> DatasetDict | Dataset:
        return ds.filter(lambda row: len(row['text']) > 0)

    def __getitem__(self, index: int) -> TextRow:
        return cast(TextRow, self.ds[index])

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
       return self.ds[index]['text']