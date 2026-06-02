from abc import abstractmethod
import hashlib
from sys import stderr
from typing import Any, Sequence, Sized
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from mirror.util import _ds_cache_path_context
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor


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
    preprocessor: MirrorPreprocessor[RawT, ProcessedT, Any],
) -> Sequence[ProcessedT]:
    def mappable_preprocessor_function(row: dict) -> dict:
        return {"input_ids": preprocessor.preprocess_example(dataset.to_row_type(row))}

    new_fingerprint: str | None = None
    if preprocessor.fingerprint is not None:
        combined = f"{dataset.ds._fingerprint}:{preprocessor.fingerprint}"
        new_fingerprint = hashlib.md5(combined.encode()).hexdigest()

    with _ds_cache_path_context():
        mapped = dataset.ds.map(
            mappable_preprocessor_function,
            new_fingerprint = new_fingerprint,
        )

    print("Preprocessing complete.", file=stderr)
    mapped.set_format(type="torch", columns=["input_ids"])

    return mapped["input_ids"]
