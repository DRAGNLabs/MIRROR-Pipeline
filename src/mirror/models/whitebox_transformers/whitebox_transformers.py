from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, cast

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
 
"""
The following types are assumptions and may need to be edited down the line
"""
RawHFOutputTypes = CausalLMOutputWithPast | CausalLMOutputWithCrossAttentions
Logits = torch.FloatTensor
LossPresent = torch.FloatTensor
HiddenStatesPresent = List[torch.FloatTensor]
AttentionsPresent = List[torch.FloatTensor]

@dataclass
class WhiteboxTransformerOutput[LossT: LossPresent | None]:
    loss: LossT


class WhiteboxTransformer[ConfigT, BatchT](ABC):
    @abstractmethod
    def fresh_config(self) -> ConfigT: pass

    @abstractmethod
    def include_loss(self, config: ConfigT, labels: torch.LongTensor) -> ConfigT: pass

    @abstractmethod
    def run(self, batch: BatchT, config: ConfigT) -> WhiteboxTransformerOutput[LossPresent | None]: pass


@dataclass(frozen=True)
class WhiteboxTransformerExecutor[LossT: LossPresent | None, ConfigT, BatchT, TransformerT: WhiteboxTransformer](ABC):
    config: ConfigT
    transformer: TransformerT

    def execute(self, batch: BatchT) -> WhiteboxTransformerOutput[LossT]:
        return cast(WhiteboxTransformerOutput[LossT], self.transformer.run(batch, self.config))

    def include_loss(self, labels: torch.LongTensor) -> \
            WhiteboxTransformerExecutor[LossPresent, ConfigT, BatchT, TransformerT]:
        return self._with_config(self.transformer.include_loss(self.config, labels))
    
    def _with_config(self, config: ConfigT):
        return WhiteboxTransformerExecutor(config=config, transformer=self.transformer)

    @classmethod
    def fresh[FreshConfigT, FreshBatchT](
        cls, 
        transformer: WhiteboxTransformer[FreshConfigT, FreshBatchT]
    ) -> WhiteboxTransformerExecutor[
        None,
        FreshConfigT,
        FreshBatchT,
        WhiteboxTransformer[FreshConfigT, FreshBatchT]
    ]:
        return WhiteboxTransformerExecutor(config=transformer.fresh_config(), transformer=transformer)
