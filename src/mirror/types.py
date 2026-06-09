from dataclasses import dataclass
from typing import TypedDict

from jaxtyping import Float, Int
from torch import Tensor


class TextRow(TypedDict):
    text: str


class TextLabelRow(TextRow):
    label: str


class PromptResponseRow(TypedDict):
    prompt: str
    response: str


TokenTensor = list[int]
AttentionMask = Int[Tensor, "T"]

TokenBatch = Int[Tensor, "b t"]
AttentionMaskBatch = Int[Tensor, "b t"]
LabelsBatch = Int[Tensor, "b t"]
StandardBatch = tuple[TokenBatch, AttentionMaskBatch, LabelsBatch]
Loss = Float[Tensor, ""]


class LabeledTokens(TypedDict):
    input_ids: list[int]
    labels: list[int]


IGNORE_ID = -100


@dataclass
class TrainStepOutput[ModelOutputT]:
    loss: Tensor
    output: ModelOutputT
