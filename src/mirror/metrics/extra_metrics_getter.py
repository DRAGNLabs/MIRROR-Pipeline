from abc import ABC, abstractmethod

from lightning import Fabric

from mirror.models.mirror_model import MirrorModel


class ExtraMetricsGetter(ABC):
    @abstractmethod
    def get_metrics(self, model: MirrorModel, fabric: Fabric) -> dict:
        """Must be called on every rank, even ranks that won't log the result.
        Implementations may invoke collectives (e.g. fabric.all_reduce); skipping
        the call on non-zero ranks will deadlock NCCL.
        """
        ...
