import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from typing import Literal, cast

from mirror.models.whitebox_transformers.whitebox_transformers import WhiteboxTransformerExecutor
from mirror.models.inference_model import InferenceModel
from mirror.models.trainable_model import TrainableModel
from mirror.models.model_util import build_causal_lm
from mirror.models.configuration_llama import LlamaConfig
from mirror.formatters.mirror_llama_formatter import MirrorLlamaFormatter
from mirror.types import LabeledTokens, Loss, StandardBatch, TextRow


class MirrorLlamaModel(
    TrainableModel[TextRow, LabeledTokens, StandardBatch],
    InferenceModel[TextRow, LabeledTokens, StandardBatch, torch.Tensor],
):
    def __init__(
        self,
        initialization: Literal["3.2-1B", "3.2-1B-Instruct"] | dict | LlamaConfig = "3.2-1B-Instruct",
        seed: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] | list[float] = (0.9, 0.999),
    ) -> None:
        super().__init__()
        self._formatter = MirrorLlamaFormatter()
        if isinstance(initialization, dict):
            initialization = LlamaConfig(**initialization)
        if isinstance(initialization, LlamaConfig):
            if seed is not None:
                torch.manual_seed(seed)
            self._hf_model = cast(LlamaForCausalLM, AutoModelForCausalLM.from_config(initialization))
        else:
            hf_model_name = f"meta-llama/Llama-{initialization}"
            self._hf_model = cast(LlamaForCausalLM, build_causal_lm(hf_model_name, weights="pretrained"))
        self._lr = lr
        self._weight_decay = weight_decay
        self._betas: tuple[float, float] = (betas[0], betas[1])

    @property
    def hf_model(self) -> LlamaForCausalLM:
        return self._hf_model

    @property
    def formatter(self) -> MirrorLlamaFormatter:
        return self._formatter

    def forward(self, batch: StandardBatch) -> torch.Tensor:
        input_ids, attention_mask, _ = batch
        return WhiteboxTransformerExecutor.fresh(self).execute((input_ids, attention_mask)).logits

    def training_step(self, batch: StandardBatch) -> Loss:
        input_ids, attention_mask, labels = batch
        return WhiteboxTransformerExecutor.fresh(self).include_loss(labels).execute((input_ids, attention_mask)).loss

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
            betas=self._betas,
        )

    def mlp_modules(self) -> list[nn.Module]:
        return [cast(nn.Module, layer.mlp) for layer in self._hf_model.model.layers]
