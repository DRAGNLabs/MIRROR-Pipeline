import torch

from mirror.tokenizers.mirror_tokenizer import MirrorTokenizer
from mirror.tokenizers.tokenizer_util import load_hf_tokenizer_from_cache_or_download
from mirror.util import get_device


class MirrorGPTTokenizer(MirrorTokenizer):
    def __init__(self, hf_model_name: str) -> None:
        self._hf_model_name = hf_model_name
        self._tokenizer = load_hf_tokenizer_from_cache_or_download(hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def tokenization_id(self):
        return self._hf_model_name

    def encode(self, text):
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        if len(ids) < 2:
            eos = self._tokenizer.eos_token_id
            eos = int(eos)
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos] # GPT causal LM loss shifts labels by 1, so seq_len=1 produces zero training targets
        return torch.tensor(ids, device=get_device(), dtype=torch.long)

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    @property
    def pad_token_id(self):
        return int(self._tokenizer.pad_token_id)
