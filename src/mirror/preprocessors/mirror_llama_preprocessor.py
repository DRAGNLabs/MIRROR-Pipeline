from typing import cast

from transformers import PreTrainedTokenizerBase

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import load_hf_tokenizer
from mirror.types import TokenBatch, AttentionMaskBatch
from mirror.util import get_device
from mirror.row_types import TextRow

class MirrorLlamaPreprocessor(
    MirrorPreprocessor[TextRow, list[int], tuple[TokenBatch, AttentionMaskBatch]]
):
    def __init__(self) -> None:
        self._hf_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(self._hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def preprocess_example(self, example: TextRow) -> list[int]:
        ids = self._tokenizer.encode(example['text'], add_special_tokens=True)
        if len(ids) < 2:
            eos = self._tokenizer.eos_token_id
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
        return cast(list[int], ids)

    def collate(self, examples: list[list[int]]) -> tuple[TokenBatch, AttentionMaskBatch]:
        device = get_device()
        batch = self._tokenizer.pad({"input_ids": examples}, padding=True, return_tensors="pt").to(device)
        tokens = cast(TokenBatch, batch["input_ids"])
        attention_mask = cast(AttentionMaskBatch, batch["attention_mask"])
        return tokens, attention_mask

    @property
    def pad_token_id(self) -> int:
        return cast(int, self._tokenizer.pad_token_id)
