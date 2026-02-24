import torch.optim as optim
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from typing import Literal, cast

from mirror.models.hf_mirror_model import HFMirrorModel
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.models.configuration_llama import LlamaConfig
from mirror.preprocessors.mirror_llama_preprocessor import MirrorLlamaPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.row_types import TextRow

class MirrorLlamaModel(HFMirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch], LlamaForCausalLM]):
    def __init__(
        self,
        initialization: Literal["3.2-1B", "3.2-1B-Instruct"] | LlamaConfig = "3.2-1B-Instruct"
    ) -> None:
        super().__init__()
        default_preprocessor_hf_name = "meta-llama/Llama-3.2-1B-Instruct"

        if isinstance(initialization, LlamaConfig):
            self.hf_model = cast(LlamaForCausalLM, AutoModelForCausalLM.from_config(initialization))
            self._preprocessor = MirrorLlamaPreprocessor(default_preprocessor_hf_name)
        else:
            hf_model_name = f"meta-llama/Llama-{initialization}"
            self.hf_model = cast(LlamaForCausalLM, build_causal_lm(hf_model_name, weights="pretrained"))
            self._preprocessor = MirrorLlamaPreprocessor(hf_model_name)

    @property
    def preprocessor(self) -> MirrorLlamaPreprocessor:
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
