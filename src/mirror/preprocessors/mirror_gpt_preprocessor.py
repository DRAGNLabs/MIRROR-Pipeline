from typing import cast

import torch
from transformers import PreTrainedTokenizerBase

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import load_hf_tokenizer
from mirror.types import TokenBatch, AttentionMaskBatch
from mirror.util import get_device
from mirror.row_types import TextRow



class MirrorGPTPreprocessor(
    MirrorPreprocessor[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]
):
    def __init__(self) -> None:
        self._hf_model_name = "openai-community/gpt2"
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(self._hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def preprocess_example(self, example: TextRow) -> list[int]:
        ids = self._tokenizer.encode(example['text'], add_special_tokens=True)
        if len(ids) < 2:
            eos = self._tokenizer.eos_token_id
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos] # GPT causal LM loss shifts labels by 1, so seq_len=1 produces zero training targets
        return cast(list[int], ids)
    
    def collate(self, examples: list[list[int]]) -> tuple[TokenBatch, AttentionMaskBatch]:
        device = get_device()
        batch = self._tokenizer.pad({"input_ids": examples}, padding=True, return_tensors="pt")
        tensors = cast(dict[str, torch.Tensor], batch)
        tokens = cast(TokenBatch, tensors["input_ids"].to(device))
        attention_mask = cast(AttentionMaskBatch, tensors["attention_mask"].to(device))
        return tokens, attention_mask

    @property
    def pad_token_id(self) -> int:
        return cast(int, self._tokenizer.pad_token_id)
