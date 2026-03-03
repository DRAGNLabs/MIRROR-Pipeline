from typing import Literal, cast

from mirror.datasets.dataset_util import load_hf_dataset
from mirror.datasets.mirror_dataset import MirrorDataset
from datasets import Dataset, DatasetDict
from mirror.row_types import TextRow

hf_dataset_path = 'stanfordnlp/imdb'


class ImdbDataset(MirrorDataset[TextRow]):
    def __init__(
        self,
        head: int | None = None,
        split: Literal['train'] | Literal['test'] | Literal['unsupervised'] = 'train',
    ):
        """
        Args:
            head: how many examples to include. None includes the whole split.
            split: which dataset split to use. 'unsupervised' is the union of
                'train' and 'test'
        """

        super().__init__()
        self.ds: Dataset = cast(DatasetDict, load_hf_dataset(
            hf_dataset_path,
        ))[split]

        if head: 
            self.ds = self.ds.select(range(head))

    @property
    def dataset_id(self) -> str:
        return hf_dataset_path

    def __getitem__(self, index: int) -> TextRow:
        return cast(TextRow, self.ds[index])

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
       return self.ds['text'][index]
