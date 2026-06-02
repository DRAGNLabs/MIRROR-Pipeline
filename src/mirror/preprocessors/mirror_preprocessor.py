from abc import ABC, abstractmethod


class MirrorPreprocessor[RawT, ProcessedT, BatchT](ABC):
    @property
    @abstractmethod
    def fingerprint(self) -> str | None:
        pass

    @abstractmethod
    def preprocess_example(self, example: RawT) -> ProcessedT:
        pass

    @abstractmethod
    def collate(self, examples: list[ProcessedT]) -> BatchT:
        pass


class DillablePreprocessorMixin:
    """Mixin for preprocessors that dill can fingerprint automatically."""
    @property
    def fingerprint(self) -> str | None:
        return None

