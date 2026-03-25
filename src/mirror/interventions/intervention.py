from abc import ABC, abstractmethod
from mirror.models.mirror_model import MirrorModel


class Intervention(ABC):
    @abstractmethod
    def transform(self, model: MirrorModel) -> MirrorModel:
        pass
