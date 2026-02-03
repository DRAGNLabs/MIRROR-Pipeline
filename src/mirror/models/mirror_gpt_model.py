from dotenv import load_dotenv
# import os
import torch
import torch.optim as optim
import torch.nn as nn

from huggingface_hub import login
from transformers import AutoModelForCausalLM

from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import load_hf_model_from_cache_or_download
from mirror.tokenizers.mirror_gpt_tokenizer import MirrorGPTTokenizer
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.util import get_device, pad_to_longest


hf_model_name = "openai-community/gpt2"

# load_dotenv(".ENV")
# hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
# login(token=hf_token)

class MirrorGPTModel(MirrorModel[str, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]):
    def __init__(self) -> None:
        super().__init__()
        self.model = load_hf_model_from_cache_or_download(
            hf_model_name,
            model_cls=AutoModelForCausalLM,
        )
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._tokenizer = MirrorGPTTokenizer(hf_model_name)

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
