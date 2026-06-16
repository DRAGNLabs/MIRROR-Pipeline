from typing import Any, Mapping

from torch import nn

from mirror.formatters.has_formatter import HasFormatter
from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer


class InferenceModel[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT, ModelOutputT](
    HasFormatter[RawT, FormattedT, BatchT],
    HFWhiteboxTransformer,
    nn.Module,
):
    pass
