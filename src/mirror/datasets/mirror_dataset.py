from abc import abstractmethod
from sys import stderr
from typing import Callable, Sequence, Sized
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from mirror.util import _ds_cache_path_context


class MirrorDataset[RawT](Dataset[RawT], Sized):
    @property
    @abstractmethod
    def ds(self) -> HFDataset:
        pass

    @abstractmethod
    def to_row_type(self, ds_row: dict) -> RawT:
        pass

    def __getitem__(self, index: int) -> RawT:
        return self.to_row_type(self.ds[index])


def preprocess_dataset[RawT, ProcessedT](
    dataset: MirrorDataset[RawT],
    preprocessor_function: Callable[[RawT], Sequence[ProcessedT]],
    num_nodes: int,
) -> Sequence[ProcessedT]:
    column_names = dataset.ds.column_names

    def mappable_preprocessor_function(batch: dict) -> dict:
        n = len(next(iter(batch.values())))
        outputs = []
        for i in range(n):
            row = {k: batch[k][i] for k in column_names}
            outputs.extend(preprocessor_function(dataset.to_row_type(row)))
        return {"input_ids": outputs}

    with _ds_cache_path_context():
        mapped = dataset.ds.map(
            mappable_preprocessor_function,
            num_proc=num_nodes,
            batched=True,
            remove_columns=column_names,
        )

    print("Preprocessing complete.", file=stderr)
    mapped.set_format(type="torch", columns=["input_ids"])

    return mapped["input_ids"]
