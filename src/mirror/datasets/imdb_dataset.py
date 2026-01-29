from typing import Literal, Sequence

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.util import load_hf_from_cache_or_download

hf_dataset_path = 'stanfordnlp/imdb'


class ImdbDataset(MirrorDataset):
    def __init__(
        self,
        head: int | None = None,
        split: Literal['train'] | Literal['test'] | Literal['unsupervised'] = 'train',
        preprocess: bool = False,
    ):
        """
        Args:
            head: how many examples to include. None includes the whole split.
            split: which dataset split to use. 'unsupervised' is the union of
                'train' and 'test'
        """

        super().__init__()
        ds = load_hf_from_cache_or_download(hf_dataset_path)

        self.ds = ds       
        self.head = head
        self.split = split
        self.preprocess_ = preprocess

    @property
    def dataset_id(self) -> str:
        return hf_dataset_path

    def __getitem__(self, index: int):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
