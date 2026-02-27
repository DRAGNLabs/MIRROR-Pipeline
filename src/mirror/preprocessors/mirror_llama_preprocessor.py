import torch
from transformers import PreTrainedTokenizerBase

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import load_hf_tokenizer
from mirror.types import TokenTensor, TokenBatch, AttentionMaskBatch
from mirror.util import get_device, pad_to_longest
from mirror.row_types import TextRow

class MirrorLlamaPreprocessor(MirrorPreprocessor):
    def __init__(self, hf_model_name: str) -> None:
        self._hf_model_name = hf_model_name
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def encode(self, text: str) -> TokenTensor:
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        if len(ids) < 2:
            eos = self._tokenizer.eos_token_id
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
        return torch.tensor(ids, device=get_device(), dtype=torch.long)

    def preprocess_example(self, example: TextRow) -> TokenTensor:
        return self.encode(example['text'])

    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return pad_to_longest(examples, pad_token=self.pad_token_id)

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id
