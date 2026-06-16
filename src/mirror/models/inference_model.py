from abc import abstractmethod
from typing import Any, Mapping

from torch import nn

from mirror.formatters.has_formatter import HasFormatter


class InferenceModel[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT, ModelOutputT](
    HasFormatter[RawT, FormattedT, BatchT],
    nn.Module,
):
    @abstractmethod
    def forward(self, batch: BatchT) -> ModelOutputT:
        pass
