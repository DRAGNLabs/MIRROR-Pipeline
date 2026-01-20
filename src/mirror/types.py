from jaxtyping import Int, Float
from torch import Tensor
from typing import TypeVar

TokenTensor = Int[Tensor, "T"]
AttentionMask = Int[Tensor, "T"]

TokenBatch = Int[Tensor, "b t"]
AttentionMaskBatch = Int[Tensor, "b t"]
Loss = Float[Tensor, ""]

RawT = TypeVar("RawT", covariant=True)
ProcessedT = TypeVar("ProcessedT", covariant=True)
BatchT = TypeVar("BatchT", covariant=True)