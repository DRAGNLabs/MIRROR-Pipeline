import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from typing import Literal, cast


from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.models.whitebox_transformers.whitebox_transformers import WhiteboxTransformerExecutor
from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.models.configuration_llama import LlamaConfig
from mirror.preprocessors.mirror_llama_preprocessor import MirrorLlamaPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.row_types import TextRow

class MirrorLlamaModel(
    MirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]],
    HFWhiteboxTransformer
):
    def __init__(
        self,
        initialization: Literal["3.2-1B", "3.2-1B-Instruct"] | LlamaConfig = "3.2-1B-Instruct",
    ) -> None:
        super().__init__()
        self._preprocessor = MirrorLlamaPreprocessor()
        if isinstance(initialization, LlamaConfig):
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
   
    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_ID) 
        labels = cast(torch.LongTensor, labels)

        output = WhiteboxTransformerExecutor.fresh(self).include_loss(labels).execute(batch)
        return output.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
