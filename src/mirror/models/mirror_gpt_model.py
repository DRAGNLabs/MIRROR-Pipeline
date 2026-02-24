import torch.optim as optim
from typing import Literal, cast

from transformers import GPT2LMHeadModel

from mirror.models.hf_mirror_model import HFMirrorModel
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.preprocessors.mirror_gpt_preprocessor import MirrorGPTPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.row_types import TextRow


hf_model_name = "openai-community/gpt2"

class MirrorGPTModel(HFMirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch], GPT2LMHeadModel]):
    def __init__(self, weights: Literal["pretrained", "random"] = "pretrained") -> None:
        super().__init__()
        self.hf_model = cast(GPT2LMHeadModel, build_causal_lm(hf_model_name, weights))
        self._preprocessor = MirrorGPTPreprocessor()

    @property
    def preprocessor(self) -> MirrorGPTPreprocessor:
        return self._preprocessor

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_ID) 

        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
