import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from typing import Literal, Tuple, TypedDict, cast

from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from transformers.utils.generic import TransformersKwargs

from mirror.models.hf_model_utils.model_output_extraction import HFTransformerInput, fresh_executor
from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.models.configuration_llama import LlamaConfig
from mirror.preprocessors.mirror_llama_preprocessor import MirrorLlamaPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.row_types import TextRow

class LlamaDict(TypedDict):
    labels: torch.LongTensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

class MirrorLlamaModel(MirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]):
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
        labels = cast(torch.LongTensor, labels)
        
        def run(kwargs: HFTransformerInput):
            return self.hf_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs 
            )

        output = fresh_executor().include_loss(labels).execute(run)
        return output.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
