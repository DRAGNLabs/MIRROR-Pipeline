from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.types import PromptResponseRow, TextRow


class CsvDataset(MirrorDataset[TextRow]):
    """
    Loads a CSV and flattens each row to a single text string via a Python
    format-string template referencing column names. Example:

        CsvDataset(file_path, template="{text}")  # single-column CSV
        CsvDataset(file_path, template="Q: {query}\\nA: {response}")
    """

    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
        self,
        file_path: str | Path,
        template: str = "{text}",
        head: int | None = None,
    ) -> None:
        super().__init__()
        self._template = template
        self._ds = cast(Dataset, load_dataset("csv", data_files=str(file_path), split="train"))
        self._ds = self._ds.filter(lambda row: len(self._render(row)) > 0)
        if head:
            self._ds = self._ds.select(range(head))

    def _render(self, row: dict) -> str:
        return self._template.format(**{k: ("" if v is None else v) for k, v in row.items()})

    def to_row_type(self, ds_row: dict) -> TextRow:
        return TextRow(text=self._render(ds_row))

    def __len__(self) -> int:
        return len(self.ds)


class CsvSftDataset(MirrorDataset[PromptResponseRow]):
    """
    Loads a CSV for supervised fine-tuning. Each row produces a
    PromptResponseRow by rendering separate prompt and response templates
    against the CSV columns. Example for a (query, response, sources) CSV:

        CsvSftDataset(
            file_path,
            prompt_template="Query: {query}\\nSources: {sources}\\nResponse: ",
            response_template="{response}",
        )
    """

    @property
    def ds(self) -> Dataset:
        return self._ds

    def __init__(
        self,
        file_path: str | Path,
        prompt_template: str,
        response_template: str,
        head: int | None = None,
    ) -> None:
        super().__init__()
        self._prompt_template = prompt_template
        self._response_template = response_template
        self._ds = cast(Dataset, load_dataset("csv", data_files=str(file_path), split="train"))
        self._ds = self._ds.filter(lambda row: len(self._render_response(row)) > 0)
        if head:
            self._ds = self._ds.select(range(head))

    def _render(self, template: str, row: dict) -> str:
        return template.format(**{k: ("" if v is None else v) for k, v in row.items()})

    def _render_response(self, row: dict) -> str:
        return self._render(self._response_template, row)

    def to_row_type(self, ds_row: dict) -> PromptResponseRow:
        return PromptResponseRow(
            prompt=self._render(self._prompt_template, ds_row),
            response=self._render_response(ds_row),
        )

    def __len__(self) -> int:
        return len(self.ds)
