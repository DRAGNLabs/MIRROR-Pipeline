from typing import cast

import torch
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.util import _ds_cache_path_context, get_device
from mirror.types import AttentionMaskBatch, TextRow, TokenBatch, TokenRow

class PlaceholderPreprocessor(
    MirrorPreprocessor[TextRow, TokenRow, tuple[TokenBatch, AttentionMaskBatch]]
):
    def format_data(self, data_source: MirrorDataset[TextRow]) -> TypedDataset[TokenRow]:
        def to_tokens(row: TextRow) -> TokenRow:
            return {"input_ids": [1, 2, 3, 4]}

        with _ds_cache_path_context():
            return data_source.ds.map(to_tokens, remove_columns=list(data_source.ds.columns))

    def collate(self, examples: list[TokenRow]) -> tuple[TokenBatch, AttentionMaskBatch]:
        tokens = cast(TokenBatch, torch.tensor([[1, 2, 3, 4]] * len(examples), device=get_device()))
        attention_mask = cast(AttentionMaskBatch, torch.ones(len(examples), 4, dtype=torch.long, device=get_device()))
        return tokens, attention_mask

    @property
    def pad_token_id(self):
        return -1
