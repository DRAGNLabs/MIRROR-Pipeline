from typing import cast

from transformers import PreTrainedTokenizerBase

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import collate_tokens, load_hf_tokenizer
from mirror.types import AttentionMaskBatch, TextRow, TokenBatch, TokenRow

class MirrorLlamaPreprocessor(
    MirrorPreprocessor[TextRow, TokenRow, tuple[TokenBatch, AttentionMaskBatch]]
):
    def __init__(self, max_length: int | None = 2048) -> None:
        self._hf_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(self._hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._max_length = max_length

    def preprocess_example(self, example: TextRow) -> TokenRow:
        ids = self._tokenizer.encode(
            example['text'],
            add_special_tokens=True,
            max_length=self._max_length,
            truncation=self._max_length is not None,
        )
        if len(ids) < 2:
            eos = self._tokenizer.eos_token_id
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
        return {"input_ids": cast(list[int], ids)}

    def collate(self, examples: list[TokenRow]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return collate_tokens(self._tokenizer, [e["input_ids"] for e in examples])

    @property
    def pad_token_id(self) -> int:
        return int(self._tokenizer.pad_token_id)  # type: ignore[arg-type]
