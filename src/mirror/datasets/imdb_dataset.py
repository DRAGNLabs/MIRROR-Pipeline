from typing import Literal, Sequence

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.util import load_hf_from_cache_or_download
from mirror.row_types import TextRow

hf_dataset_path = 'stanfordnlp/imdb'


class ImdbDataset(MirrorDataset[str]):
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
        
        self.examples: Sequence[str] = ds[split]['text']  # pyright: ignore
        if head:
            self.examples = self.examples[:head]
        self.ds = load_hf_from_cache_or_download(
            hf_dataset_path,
        )[split]

        if head: 
            self.ds = self.ds.select(range(head))

    @property
    def dataset_id(self) -> str:
        return hf_dataset_path

    def __getitem__(self, index: int) -> TextRow:
        return self.ds[index]

    def __len__(self) -> int:
        return len(self.ds)

    def item(self, index) -> str:
       return self.ds['text'][index]
