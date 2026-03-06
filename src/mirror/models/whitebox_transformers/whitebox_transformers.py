from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
 
"""
The following types are assumptions and may need to be edited down the line
"""
type RawHFOutputTypes = CausalLMOutputWithPast | CausalLMOutputWithCrossAttentions
type Logits = torch.FloatTensor
type LossPresent = torch.FloatTensor
type HiddenStatesPresent = list[torch.FloatTensor]
type AttentionsPresent = list[torch.FloatTensor]

@dataclass
class WhiteboxTransformerOutput[LossT: LossPresent | None, HiddenStatesT: HiddenStatesPresent | None, AttentionsT: AttentionsPresent | None]:
    logits: Logits
    loss: LossT
    hidden_states: HiddenStatesT
    attentions: AttentionsT

AnyWhiteboxTransformerOutput = WhiteboxTransformerOutput[LossPresent | None, HiddenStatesPresent | None, AttentionsPresent | None]

class WhiteboxTransformer[ConfigT, BatchT](ABC):
    """
    Implementing this interface allows a model to be used with `WhiteboxTransformerExecutor`.
    (See `WhiteboxTransformerExecutor` for why you might want this).

    Note that the implementation *must* follow the invariants described on the methods, or
    you may run into runtime type errors that the type checker won't catch.

    The central object is the `config` object. It describes a model execution configuration,
    including whether the execution should return the hidden_states, loss, etc. It doesn't
    matter *what* the object is, as long as it can be used in `run`.
    """

    @abstractmethod
    def fresh_config(self) -> ConfigT:
        """
        Returns a "fresh" config that indicates that none of the optional items should be returned.
        """
        pass

    @abstractmethod
    def include_loss(self, config: ConfigT, labels: torch.LongTensor) -> ConfigT:
        """
        Returns a modified config that indicates loss should be returned (calculated
        with respect to the labels).
        """
        pass

    @abstractmethod
    def include_hidden_states(self, config: ConfigT) -> ConfigT:
        """
        Returns a modified config that indicates hidden states should be returned.
        """
        pass

    @abstractmethod
    def include_attentions(self, config: ConfigT) -> ConfigT:
        """
        Returns a modified config that indicates attentions should be returned.
        """
        pass

    @abstractmethod
    def run(self, batch: BatchT, config: ConfigT) -> AnyWhiteboxTransformerOutput:
        """
        While the return type is `AnyWhiteboxTransformerOutput`, the output returned by this
        method *must* respect the `config` passed in. In other words, if the config says
        hidden states should be included, the output *must* include hidden states (not `None`).
        """
        pass


@dataclass(frozen=True)
class WhiteboxTransformerExecutor[LossT: LossPresent | None, HiddenStatesT: HiddenStatesPresent | None, AttentionsT: AttentionsPresent | None, ConfigT, BatchT, TransformerT: WhiteboxTransformer](ABC):
    """
    This class aims to solve the problem that return types of models from various libraries
    tend to return outputs with optional members. For instance, `loss` on a model often has a
    type something like `torch.FloatTensor | None`. This can get annoying with type checkers,
    since in most cases, it is statically known whether loss will be computed or not, but
    the type checker will still yell at you unless you explicitly check for `None`.

    This class accomplishes that goal using a fluent interface. Use `.fresh(model)` passing it
    your model (that implements `WhiteboxTransformer`), then fluently call any of the `.include*`
    methods to add items to the output. When you're ready, call `.execute(batch)`, which will
    return an output that includes all of the output items in a way that the type checker knows
    about.
    """

    config: ConfigT
    transformer: TransformerT

    def execute(self, batch: BatchT) -> WhiteboxTransformerOutput[LossT, HiddenStatesT, AttentionsT]:
        return cast(
            WhiteboxTransformerOutput[LossT, HiddenStatesT, AttentionsT],
            self.transformer.run(batch, self.config)
        )

    def include_loss(self, labels: torch.LongTensor) -> \
            WhiteboxTransformerExecutor[LossPresent, HiddenStatesT, AttentionsT, ConfigT, BatchT, TransformerT]:
        return self._with_config(self.transformer.include_loss(self.config, labels))
    
    def include_hidden_states(self) -> \
            WhiteboxTransformerExecutor[LossT, HiddenStatesPresent, AttentionsT, ConfigT, BatchT, TransformerT]:
        return self._with_config(self.transformer.include_hidden_states(self.config))
     
    def include_attentions(self) -> \
            WhiteboxTransformerExecutor[LossT, HiddenStatesT, AttentionsPresent, ConfigT, BatchT, TransformerT]:
        return self._with_config(self.transformer.include_attentions(self.config))

    def _with_config(self, config: ConfigT):
        return WhiteboxTransformerExecutor(config=config, transformer=self.transformer)

    @classmethod
    def fresh[FreshConfigT, FreshBatchT](
        cls, 
        transformer: WhiteboxTransformer[FreshConfigT, FreshBatchT]
    ) -> WhiteboxTransformerExecutor[
        None,
        None,
        None,
        FreshConfigT,
        FreshBatchT,
        WhiteboxTransformer[FreshConfigT, FreshBatchT]
    ]:
        return WhiteboxTransformerExecutor(config=transformer.fresh_config(), transformer=transformer)
