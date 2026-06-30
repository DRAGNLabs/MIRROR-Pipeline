from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import PromptResponseRow
from mirror.util import _ds_cache_path_context


class EchoDataset(MirrorDataset[PromptResponseRow]):
    """
    Sanity-check dataset for instruction fine-tuning. Reads a single column
    from a CSV and sets response = prompt, so the model must learn to echo
    its input. If instruction fine-tuning can't solve this task, something
    is broken in the pipeline.

        EchoDataset(file_path)                        # uses "query" column
        EchoDataset(file_path, prompt_column="text")  # custom column name
    """

    @property
    def ds(self) -> TypedDataset[PromptResponseRow]:
        return self._ds

    def __init__(
        self,
        file_path: str | Path,
        prompt_column: str = "query",
        skip: int | None = None,
        head: int | None = None,
    ) -> None:
        super().__init__()
        col = prompt_column

        def render(row: dict) -> PromptResponseRow:
            text = "" if row[col] is None else str(row[col])
            return PromptResponseRow(prompt=text, response=text)

        raw = cast(Dataset, load_dataset("csv", data_files=str(file_path), split="train"))
        if skip:
            raw = raw.select(range(skip, len(raw)))
        with _ds_cache_path_context():
            ds = TypedDataset[PromptResponseRow](raw.map(render, remove_columns=raw.column_names))
            ds = ds.filter(lambda row: len(row["prompt"]) > 0)
            if head:
                ds = ds.take(head)
        self._ds = ds

    def __len__(self) -> int:
        return len(self.ds)
