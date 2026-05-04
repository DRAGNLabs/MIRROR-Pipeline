from dataclasses import dataclass
from typing import TypedDict

from jaxtyping import Float, Int
from torch import Tensor


class TextRow(TypedDict):
    text: str


class TextLabelRow(TextRow):
    label: str


TokenTensor = list[int]
AttentionMask = Int[Tensor, "T"]

TokenBatch = Int[Tensor, "b t"]
AttentionMaskBatch = Int[Tensor, "b t"]
Loss = Float[Tensor, ""]


@dataclass
class TrainStepOutput[ModelOutputT]:
    loss: Tensor
    output: ModelOutputT
