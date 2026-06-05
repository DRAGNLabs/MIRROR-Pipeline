import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from typing import Literal, cast


from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.models.whitebox_transformers.whitebox_transformers import WhiteboxTransformerExecutor
from mirror.models.inference_model import InferenceModel
from mirror.models.trainable_model import TrainableModel
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.models.configuration_llama import LlamaConfig
from mirror.preprocessors.mirror_llama_preprocessor import MirrorLlamaPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TextRow, TokenBatch, TokenTensor


class MirrorLlamaModel(
    TrainableModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]],
    InferenceModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch], torch.Tensor],
    HFWhiteboxTransformer,
):
    def __init__(
        self,
        initialization: Literal["3.2-1B", "3.2-1B-Instruct"] | LlamaConfig = "3.2-1B-Instruct",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._preprocessor = MirrorLlamaPreprocessor()
        if isinstance(initialization, LlamaConfig):
            if seed is not None:
                torch.manual_seed(seed)
            self._hf_model = cast(LlamaForCausalLM, AutoModelForCausalLM.from_config(initialization))
        else:
            hf_model_name = f"meta-llama/Llama-{initialization}"
            self._hf_model = cast(LlamaForCausalLM, build_causal_lm(hf_model_name, weights="pretrained"))

    @property
    def hf_model(self) -> LlamaForCausalLM:
        return self._hf_model

    @property
    def preprocessor(self) -> MirrorLlamaPreprocessor:
        return self._preprocessor

    def forward(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> torch.Tensor:
        return WhiteboxTransformerExecutor.fresh(self).execute(batch).logits

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_ID)
        labels = cast(torch.Tensor, labels)

        return WhiteboxTransformerExecutor.fresh(self).include_loss(labels).execute(batch).loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

    def mlp_modules(self) -> list[nn.Module]:
        return [cast(nn.Module, layer.mlp) for layer in self._hf_model.model.layers]
