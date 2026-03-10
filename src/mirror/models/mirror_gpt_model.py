import torch
import torch.optim as optim
from typing import Literal, Union, cast

from transformers import GPT2LMHeadModel

from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.models.whitebox_transformers.whitebox_transformers import WhiteboxTransformerExecutor
from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.preprocessors.mirror_gpt_preprocessor import MirrorGPTPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.row_types import TextRow


hf_model_name = "openai-community/gpt2"

class MirrorGPTModel(
    MirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]],
    HFWhiteboxTransformer    
):
    def __init__(self, weights: Literal["pretrained", "random"] = "pretrained") -> None:
        super().__init__()
        self._hf_model = cast(GPT2LMHeadModel, build_causal_lm(hf_model_name, weights))
        self._preprocessor = MirrorGPTPreprocessor()

    @property
    def hf_model(self) -> GPT2LMHeadModel:
        return self._hf_model

    @property
    def preprocessor(self) -> MirrorGPTPreprocessor:
        return self._preprocessor

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_ID) 
        labels = cast(torch.Tensor, labels)

        output = WhiteboxTransformerExecutor.fresh(self).include_loss(labels).execute(batch)
        return output.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

def assert_union_snd[A, B](union: Union[A, B]) -> A:
    return union #pyright: ignore
