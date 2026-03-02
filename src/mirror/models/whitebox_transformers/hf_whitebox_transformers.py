from abc import abstractmethod

import torch
from transformers import PreTrainedModel
from transformers.utils.generic import TransformersKwargs

from mirror.models.whitebox_transformers.whitebox_transformers import AnyWhiteboxTransformerOutput, LossPresent, WhiteboxTransformer, WhiteboxTransformerOutput
from mirror.types import AttentionMaskBatch, TokenBatch

class HFTransformerInput(TransformersKwargs):
    labels: torch.LongTensor | None

   
HFWhiteboxTransformerConfig = HFTransformerInput

class HFWhiteboxTransformer(
    WhiteboxTransformer[HFWhiteboxTransformerConfig, tuple[TokenBatch, AttentionMaskBatch]]
):
    """
    Inheriting from this class allows a model to be used with `WhiteboxTransformerExecutor`
    (which allows for type-safe extraction of loss, hidden states, and attention)
    """

    @property
    @abstractmethod
    def hf_model(self) -> PreTrainedModel: pass

    def fresh_config(self) -> HFTransformerInput:
        return HFTransformerInput(**TransformersKwargs(), labels=None)
    
    def include_loss(self, config: HFTransformerInput, labels: TokenBatch) -> HFTransformerInput:
        return HFTransformerInput({**config, 'labels': labels})
     
    def include_hidden_states(self, config: HFTransformerInput) -> HFTransformerInput:
        return HFTransformerInput({**config, 'output_hidden_states': True})

    def include_attentions(self, config: HFTransformerInput) -> HFTransformerInput:
        return HFTransformerInput({**config, 'output_attentions': True})
    
    def run(
            self,
            batch: tuple[TokenBatch, AttentionMaskBatch],
            config: HFWhiteboxTransformerConfig
    ) -> AnyWhiteboxTransformerOutput:
        return self.hf_model(
            input_ids=batch[0],
            attention_mask=batch[1],
            **config
        )