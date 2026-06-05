from abc import ABC, abstractmethod

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor


class HasPreprocessor[RawT, ProcessedT, BatchT](ABC):
    @property
    @abstractmethod
    def preprocessor(self) -> MirrorPreprocessor[RawT, ProcessedT, BatchT]:
        pass
