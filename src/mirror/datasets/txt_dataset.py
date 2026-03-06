from pathlib import Path

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.row_types import TextRow


class TxtDataset(MirrorDataset[TextRow]):
    def __init__(
            self,
            file_path: str | Path, 
            head: int | None = None
    ):
        """
        Args:
            file_path: path to a .txt file where each line is one example.
            head: how many examples to include. None includes the whole split.
        """
        super().__init__()
        self._file_path = Path(file_path)
        self._lines: list[str] = []

        with open(self._file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if head is not None and i >= head:
                    break
                stripped = line.rstrip("\n")
                if stripped:
                    self._lines.append(stripped)

    @property
    def dataset_id(self) -> str:
        return str(self._file_path)

    def __getitem__(self, index: int) -> TextRow:
        return TextRow(text=self._lines[index])

    def __len__(self) -> int:
        return len(self._lines)
