import dask.dataframe as dd

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.util import tokenized_path


class WikitextDataset(MirrorDataset):
    def __init__(
            self,
            dir_path=tokenized_path / 'wikitext/train',
            head: int | None = None,
            max_sequence_embeddings: int = 1024,
    ):
        super().__init__(dir_path, head, max_sequence_embeddings)
        self.data = dd.read_parquet(dir_path / '*.parquet', columns='text')
        if self.head:
            self.data = self.data.head(self.head)

    def get_raw_iter(self):
        return iter(self.data)
