import torch.nn as nn
import torch.optim as optim
from typing import Literal, Union, cast

from transformers import GPT2LMHeadModel

from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.models.whitebox_transformers.whitebox_transformers import WhiteboxTransformerExecutor
from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import build_causal_lm
from mirror.formatters.mirror_gpt_formatter import MirrorGPTFormatter
from mirror.types import LabeledTokens, StandardBatch, TextRow, TrainStepOutput


hf_model_name = "openai-community/gpt2"

class MirrorGPTModel(
    MirrorModel[TextRow, LabeledTokens, StandardBatch, None],
    HFWhiteboxTransformer
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

    def training_step(self, batch: StandardBatch) -> TrainStepOutput[None]:
        input_ids, attention_mask, labels = batch
        output = WhiteboxTransformerExecutor.fresh(self).include_loss(labels).execute((input_ids, attention_mask))
        return TrainStepOutput(loss=output.loss, output=None)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

    def mlp_modules(self) -> list[nn.Module]:
        return [cast(nn.Module, block.mlp) for block in self._hf_model.transformer.h]

def assert_union_snd[A, B](union: Union[A, B]) -> A:
    return union #pyright: ignore
