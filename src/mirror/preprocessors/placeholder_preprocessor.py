from typing import cast

import torch

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.util import get_device
from mirror.types import AttentionMaskBatch, TextRow, TokenBatch, TokenTensor

class PlaceholderPreprocessor(
    MirrorPreprocessor[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]
):
    def preprocess_example(self, example: TextRow) -> TokenTensor:
        return [1, 2, 3, 4]
    
    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        tokens = cast(TokenBatch, torch.tensor([[1, 2, 3, 4]] * len(examples), device=get_device()))
        attention_mask = cast(AttentionMaskBatch, torch.ones(len(examples), 4, dtype=torch.long, device=get_device()))
        return tokens, attention_mask

    @property
    def pad_token_id(self):
        return -1
