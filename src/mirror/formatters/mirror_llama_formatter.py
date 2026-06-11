from typing import cast

from transformers import PreTrainedTokenizerBase
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.formatters.infer_friendly_preprocessor import InferFriendlyPreprocessor
from mirror.formatters.mirror_formatter import MirrorFormatter
from mirror.formatters.formatter_util import collate_tokens, load_hf_tokenizer
from mirror.types import LabeledTokens, StandardBatch, TextRow, TokenTensor
from mirror.util import _ds_cache_path_context

class MirrorLlamaFormatter(
    InferFriendlyPreprocessor,
    MirrorFormatter[TextRow, LabeledTokens, StandardBatch],
):
    def __init__(self, max_length: int | None = 2048) -> None:
        self._hf_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(self._hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._max_length = max_length

    def format_data(self, data_source: MirrorDataset[TextRow]) -> TypedDataset[LabeledTokens]:
        tokenizer = self._tokenizer
        max_length = self._max_length

        def tokenize(row: TextRow) -> LabeledTokens:
            ids = tokenizer.encode(
                row['text'],
                add_special_tokens=True,
                max_length=max_length,
                truncation=max_length is not None,
            )
            if len(ids) < 2:
                eos = tokenizer.eos_token_id
                ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
            token_ids = cast(TokenTensor, ids)
            return LabeledTokens(input_ids=(token_ids), labels=list(token_ids))

        with _ds_cache_path_context():
            return data_source.ds.map(tokenize, remove_columns=list(data_source.ds.columns))

    def collate(self, examples: list[LabeledTokens]) -> StandardBatch:
        return collate_tokens(self._tokenizer, examples)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def pad_token_id(self) -> int:
        return int(self._tokenizer.pad_token_id)  # type: ignore[arg-type]
