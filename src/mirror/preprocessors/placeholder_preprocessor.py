from typing import cast

import torch
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.util import _ds_cache_path_context, get_device
from mirror.types import AttentionMaskBatch, LabeledTokens, LabelsBatch, StandardBatch, TextRow, TokenBatch

class PlaceholderPreprocessor(
    MirrorPreprocessor[TextRow, LabeledTokens, StandardBatch]
):
    def format_data(self, data_source: MirrorDataset[TextRow]) -> TypedDataset[LabeledTokens]:
        def to_tokens(row: TextRow) -> LabeledTokens:
            ids = [1, 2, 3, 4]
            return LabeledTokens(input_ids=ids, labels=list(ids))

        with _ds_cache_path_context():
            return data_source.ds.map(to_tokens, remove_columns=list(data_source.ds.columns))

    def collate(self, examples: list[LabeledTokens]) -> StandardBatch:
        device = get_device()
        batch_size = len(examples)
        tokens = cast(TokenBatch, torch.tensor([[1, 2, 3, 4]] * batch_size, device=device))
        attention_mask = cast(AttentionMaskBatch, torch.ones(batch_size, 4, dtype=torch.long, device=device))
        labels = cast(LabelsBatch, torch.tensor([[1, 2, 3, 4]] * batch_size, device=device))
        return tokens, attention_mask, labels

    @property
    def pad_token_id(self):
        return -1
