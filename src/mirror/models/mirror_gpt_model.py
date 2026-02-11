import torch
import torch.optim as optim
import torch.nn as nn
from typing import Literal

from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import build_causal_lm, ignore_id
from mirror.tokenizers.mirror_gpt_tokenizer import MirrorGPTTokenizer
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.util import get_device, pad_to_longest


hf_model_name = "openai-community/gpt2"

class MirrorGPTModel(MirrorModel[str, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]):
    def __init__(self, weights: Literal["pretrained", "random"] = "pretrained") -> None:
        super().__init__()
        self.model = build_causal_lm(hf_model_name, weights)
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._tokenizer = MirrorGPTTokenizer()

    @property
    def tokenizer(self) -> MirrorGPTTokenizer:
        return self._tokenizer

    def preprocess_example(self, text: str) -> TokenTensor:
        return self.tokenizer.encode(text)

    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return pad_to_longest(examples, pad_token=self.tokenizer.pad_token_id)

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, ignore_id) 

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
