from abc import ABC, abstractmethod

from mirror.models.mirror_model import MirrorModel


class ExtraMetricsGetter(ABC):
    @abstractmethod
    def get_metrics(self, model: MirrorModel) -> dict:
        ...
