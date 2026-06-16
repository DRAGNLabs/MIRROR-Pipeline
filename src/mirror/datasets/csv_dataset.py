from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import PromptResponseRow, TextRow
from mirror.util import _ds_cache_path_context


class CsvDataset(MirrorDataset[TextRow]):
    """
    Loads a CSV and flattens each row to a single text string via a Python
    format-string template referencing column names. Example:

        CsvDataset(file_path, template="{text}")  # single-column CSV
        CsvDataset(file_path, template="Q: {query}\\nA: {response}")
    """

    @property
    def ds(self) -> TypedDataset[TextRow]:
        return self._ds

    def __init__(
        self,
        file_path: str | Path,
        template: str = "{text}",
        head: int | None = None,
    ) -> None:
        super().__init__()
        tmpl = template

        def render(row: dict) -> TextRow:
            return TextRow(text=tmpl.format(**{k: ("" if v is None else v) for k, v in row.items()}))

        raw = cast(Dataset, load_dataset("csv", data_files=str(file_path), split="train"))
        with _ds_cache_path_context():
            ds = TypedDataset[TextRow](raw.map(render, remove_columns=raw.column_names))
            ds = ds.filter(lambda row: len(row['text']) > 0)
            if head:
                ds = ds.take(head)
        self._ds = ds

    def __len__(self) -> int:
        return len(self.ds)


class CsvInstructDataset(MirrorDataset[PromptResponseRow]):
    """
    Loads a CSV for supervised fine-tuning. Each row produces a
    PromptResponseRow by rendering separate prompt and response templates
    against the CSV columns. The prompt is the user-message content and the
    response is the assistant-message content; role markers and special
    tokens are added later by the instruct formatter's chat template (when
    the tokenizer has one). Example for a (query, response, sources) CSV:

        CsvInstructDataset(
            file_path,
            prompt_template="{query}\\nSources: {sources}",
            response_template="{response}",
        )
    """

    @property
    def ds(self) -> TypedDataset[PromptResponseRow]:
        return self._ds

    def __init__(
        self,
        file_path: str | Path,
        prompt_template: str,
        response_template: str,
        head: int | None = None,
    ) -> None:
        super().__init__()
        ptmpl = prompt_template
        rtmpl = response_template

        def render(row: dict) -> PromptResponseRow:
            cleaned = {k: ("" if v is None else v) for k, v in row.items()}
            return PromptResponseRow(
                prompt=ptmpl.format(**cleaned),
                response=rtmpl.format(**cleaned),
            )

        raw = cast(Dataset, load_dataset("csv", data_files=str(file_path), split="train"))
        with _ds_cache_path_context():
            ds = TypedDataset[PromptResponseRow](raw.map(render, remove_columns=raw.column_names))
            ds = ds.filter(lambda row: len(row['response']) > 0)
            if head:
                ds = ds.take(head)
        self._ds = ds

    def __len__(self) -> int:
        return len(self.ds)
