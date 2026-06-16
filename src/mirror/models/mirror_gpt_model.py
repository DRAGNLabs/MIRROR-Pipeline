import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal, cast

from transformers import GPT2LMHeadModel

from mirror.models.whitebox_transformers.whitebox_transformers import WhiteboxTransformerExecutor
from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.models.inference_model import InferenceModel
from mirror.models.trainable_model import TrainableModel
from mirror.models.model_util import build_causal_lm
from mirror.formatters.mirror_gpt_formatter import MirrorGPTFormatter
from mirror.types import LabeledTokens, Loss, StandardBatch, TextRow


hf_model_name = "openai-community/gpt2"


class MirrorGPTModel(
    TrainableModel[TextRow, LabeledTokens, StandardBatch],
    InferenceModel[TextRow, LabeledTokens, StandardBatch, torch.Tensor],
    HFWhiteboxTransformer,
):
    def __init__(self, weights: Literal["pretrained", "random"] = "pretrained") -> None:
        super().__init__()
        self._hf_model = cast(GPT2LMHeadModel, build_causal_lm(hf_model_name, weights))
        self._formatter = MirrorGPTFormatter()

    @property
    def hf_model(self) -> GPT2LMHeadModel:
        return self._hf_model

    @property
    def formatter(self) -> MirrorGPTFormatter:
        return self._formatter

    def forward(self, batch: StandardBatch) -> torch.Tensor:
        input_ids, attention_mask, _ = batch
        return WhiteboxTransformerExecutor.fresh(self).execute((input_ids, attention_mask)).logits

    def training_step(self, batch: StandardBatch) -> Loss:
        input_ids, attention_mask, labels = batch
        return WhiteboxTransformerExecutor.fresh(self).include_loss(labels).execute((input_ids, attention_mask)).loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

    def mlp_modules(self) -> list[nn.Module]:
        return [cast(nn.Module, block.mlp) for block in self._hf_model.transformer.h]
