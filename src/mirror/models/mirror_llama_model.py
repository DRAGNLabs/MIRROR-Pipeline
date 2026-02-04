from dotenv import load_dotenv
# import os
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Literal

from huggingface_hub import login
from transformers import AutoModelForCausalLM

from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import load_hf_config_from_cache_or_download, load_hf_model_from_cache_or_download
from mirror.tokenizers.mirror_llama_tokenizer import MirrorLlamaTokenizer
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.util import get_device, pad_to_longest


class MirrorLlamaModel(MirrorModel[str, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]):
    def __init__(
        self,
        id: Literal["3.2-1B", "3.2-1B-Instruct"] = "3.2-1B",
        weights: Literal["pretrained", "random"] = "pretrained",
    ) -> None:
        super().__init__()
        hf_model_name = f"meta-llama/Llama-{id}"
        if weights == "pretrained":
            self.model = load_hf_model_from_cache_or_download(
                hf_model_name,
                model_cls=AutoModelForCausalLM,
            )
        elif weights == "random":
            config = load_hf_config_from_cache_or_download(hf_model_name)
            self.model = AutoModelForCausalLM.from_config(config)
        else:
            raise ValueError(f"Unknown weights option: {weights}")
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._tokenizer = MirrorLlamaTokenizer(hf_model_name)

    @property
    def tokenizer(self):
        return self._tokenizer

    def preprocess_example(self, text: str) -> TokenTensor:
        return self.tokenizer.encode(text)

    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return pad_to_longest(examples, pad_token=self.tokenizer.pad_token_id)

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100) 

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
