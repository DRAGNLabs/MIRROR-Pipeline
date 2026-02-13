import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaConfig
from typing import Literal

from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.tokenizers.mirror_llama_tokenizer import MirrorLlamaTokenizer
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.util import pad_to_longest


class MirrorLlamaModel(MirrorModel[str, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]):
    def __init__(
        self,
        initialization: Literal["3.2-1B", "3.2-1B-Instruct"] | LlamaConfig = "3.2-1B-Instruct"
    ) -> None:
        super().__init__()
        default_tokenizer_hf_name = "meta-llama/Llama-3.2-1B-Instruct"

        if isinstance(initialization, LlamaConfig):
            self.model = AutoModelForCausalLM.from_config(initialization)
            self._tokenizer = MirrorLlamaTokenizer(default_tokenizer_hf_name)
        else:
            hf_model_name = f"meta-llama/Llama-{initialization}"
            self.model = build_causal_lm(hf_model_name, weights="pretrained")
            self._tokenizer = MirrorLlamaTokenizer(hf_model_name)

    @property
    def tokenizer(self) -> MirrorLlamaTokenizer:
        return self._tokenizer

    def preprocess_example(self, text: str) -> TokenTensor:
        return self.tokenizer.encode(text)

    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return pad_to_longest(examples, pad_token=self.tokenizer.pad_token_id)

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_ID) 

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
