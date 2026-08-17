from abc import abstractmethod
from typing import Any, Mapping

from torch import nn

from mirror.formatters.infer_friendly_formatter import InferFriendlyFormatter
from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer


class InferenceModel[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT, ModelOutputT](
    HFWhiteboxTransformer,
    nn.Module,
):
    @property
    @abstractmethod
    def formatter(self) -> InferFriendlyFormatter:
        pass
