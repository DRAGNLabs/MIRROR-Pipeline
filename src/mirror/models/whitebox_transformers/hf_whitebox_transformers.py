from abc import abstractmethod

import torch
from transformers import PreTrainedModel
from transformers.utils.generic import TransformersKwargs

from mirror.models.whitebox_transformers.whitebox_transformers import LossPresent, WhiteboxTransformer, WhiteboxTransformerOutput
from mirror.types import AttentionMaskBatch, TokenBatch

class HFTransformerInput(TransformersKwargs):
    labels: torch.LongTensor | None

   
HFWhiteboxTransformerConfig = HFTransformerInput

class HFWhiteboxTransformer(
    WhiteboxTransformer[HFWhiteboxTransformerConfig, tuple[TokenBatch, AttentionMaskBatch]]
):
    @property
    @abstractmethod
    def hf_model(self) -> PreTrainedModel: pass

    def fresh_config(self) -> HFTransformerInput:
        return HFTransformerInput(**TransformersKwargs(), labels=None)
    
    def include_loss(self, config: HFTransformerInput, labels: TokenBatch) -> HFTransformerInput:
        return HFTransformerInput({**config, 'labels': labels})
    
    def run(
            self,
            batch: tuple[TokenBatch, AttentionMaskBatch],
            config: HFWhiteboxTransformerConfig
    ) -> WhiteboxTransformerOutput[LossPresent | None]:
        return self.hf_model(
            input_ids=batch[0],
            attention_mask=batch[1],
            **config
        )