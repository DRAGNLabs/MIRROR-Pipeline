from typing import Sequence

from datasets import Dataset, DatasetDict

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.util import load_hf_from_cache_or_download

hf_dataset_path = 'Salesforce/wikitext'
hf_dataset_name = 'wikitext-2-raw-v1'


class WikitextDataset(MirrorDataset):
    def __init__(
            self,
            head: int
    ):
        super().__init__()
        ds = load_hf_from_cache_or_download(
            hf_dataset_path,
            hf_dataset_name,
            self._process,  # pyright: ignore
        )
        self.examples: Sequence[str] = ds['text']  # pyright: ignore
        if head:
            self.examples = self.examples[:head]

    @property
    def dataset_id(self) -> str:
        return f'{hf_dataset_path}/{hf_dataset_name}'

    def _process(self, ds: DatasetDict) -> Dataset:
        split = ds['train']
        return split.filter(lambda row: len(row['text']) > 0)

    def __getitem__(self, index: int):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
