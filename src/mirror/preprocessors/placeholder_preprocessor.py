from typing import cast

import torch

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.util import get_device
from mirror.types import AttentionMaskBatch, LabeledTokens, LabelsBatch, TextRow, TokenBatch

class PlaceholderPreprocessor(
    MirrorPreprocessor[TextRow, LabeledTokens, tuple[TokenBatch, AttentionMaskBatch, LabelsBatch]]
):
    def preprocess_example(self, example: TextRow) -> LabeledTokens:
        ids = [1, 2, 3, 4]
        return LabeledTokens(input_ids=ids, labels=list(ids))

    def collate(self, examples: list[LabeledTokens]) -> tuple[TokenBatch, AttentionMaskBatch, LabelsBatch]:
        device = get_device()
        batch_size = len(examples)
        tokens = cast(TokenBatch, torch.tensor([[1, 2, 3, 4]] * batch_size, device=device))
        attention_mask = cast(AttentionMaskBatch, torch.ones(batch_size, 4, dtype=torch.long, device=device))
        labels = cast(LabelsBatch, torch.tensor([[1, 2, 3, 4]] * batch_size, device=device))
        return tokens, attention_mask, labels

    @property
    def pad_token_id(self):
        return -1
