import torch
from transformers import PreTrainedTokenizerBase

from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.tokenizers.tokenizer_util import load_hf_tokenizer
from mirror.types import TokenTensor
from mirror.util import get_device


class MirrorGPTTokenizer(MirrorTokenizer):
    def __init__(self) -> None:
        self._hf_model_name = "openai-community/gpt2"
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(self._hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def tokenization_id(self) -> str:
        return "MirrorGPTTokenizer"

    def encode(self, text: str) -> TokenTensor:
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        if len(ids) < 2:
            eos = self._tokenizer.eos_token_id
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos] # GPT causal LM loss shifts labels by 1, so seq_len=1 produces zero training targets
        return torch.tensor(ids, device=get_device(), dtype=torch.long)

    def decode(self, tokens: TokenTensor) -> str:
        tokens = tokens.tolist()
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id
