from abc import ABC, abstractmethod
from typing import Any, Mapping

from lightning import Fabric

from mirror.models.mirror_model import MirrorModel


class MirrorMetric[RawT: Mapping[str, Any], ProcessedT: Mapping[str, Any], BatchT, ModelOutputT](ABC):
    @abstractmethod
    def get_metrics(self, model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT], fabric: Fabric) -> dict:
        """Must be called on every rank, even ranks that won't log the result.
        Implementations may invoke collectives (e.g. fabric.all_reduce); skipping
        the call on non-zero ranks will deadlock NCCL.
        """
        ...
