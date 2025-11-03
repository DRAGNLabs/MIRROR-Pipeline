from typing import Sequence

from datasets import Dataset, DatasetDict

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.util import load_hf_from_cache_or_download

hf_dataset_path = 'stanfordnlp/imdb'


class ImdbDataset(MirrorDataset):
    def __init__(
        self,
        head: int,
    ):
        super().__init__()
        ds = load_hf_from_cache_or_download(
            hf_dataset_path,
            process=self._process,  # pyright: ignore
        )
        self.examples: Sequence[str] = ds['text']  # pyright: ignore
        if head:
            self.examples = self.examples[:head]

    @property
    def dataset_id(self) -> str:
        return hf_dataset_path

    def _process(self, ds: DatasetDict) -> Dataset:
        return ds['train']

    def __getitem__(self, index: int):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
