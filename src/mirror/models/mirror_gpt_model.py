import torch
import torch.optim as optim
from typing import Callable, Literal, Tuple, TypedDict, Union, cast

from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from mirror.models.hf_mirror_model import HFMirrorModel
from mirror.models.hf_model_utils.model_output_extraction import tuplize, with_loss
from mirror.models.model_util import build_causal_lm, IGNORE_ID
from mirror.preprocessors.mirror_gpt_preprocessor import MirrorGPTPreprocessor
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.row_types import TextRow


hf_model_name = "openai-community/gpt2"

class GptDict(TypedDict):
    labels: torch.LongTensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

class MirrorGPTModel(HFMirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch], GPT2LMHeadModel]):
    def __init__(self, weights: Literal["pretrained", "random"] = "pretrained") -> None:
        super().__init__()
        self.hf_model = cast(GPT2LMHeadModel, build_causal_lm(hf_model_name, weights))
        self._preprocessor = MirrorGPTPreprocessor()

    @property
    def preprocessor(self) -> MirrorGPTPreprocessor:
        return self._preprocessor
    
    def call_hf(self, args: GptDict) -> Tuple[CausalLMOutputWithPast]:
        return self.hf_model(**args)

    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        input_ids, attention_mask = batch
        labels = input_ids
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_ID) 
        labels = cast(torch.LongTensor, labels)

        test = with_loss(tuplize(assert_union_snd(self.hf_model.forward)))
        loss, _ = with_loss(tuplize(assert_union_snd(self.hf_model.forward)))(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters())

def assert_union_snd[**P, A, B](f: Callable[P, Union[A, B]]) -> Callable[P, A]:
    return f #pyright: ignore
