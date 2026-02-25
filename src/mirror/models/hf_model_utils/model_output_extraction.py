from __future__ import annotations
from typing import Callable, List, cast

from attr import dataclass
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from transformers.utils.generic import TransformersKwargs

"""
The following types are assumptions and made need to be edited down the line
"""
RawHFOutputTypes = CausalLMOutputWithPast | CausalLMOutputWithCrossAttentions
Logits = torch.FloatTensor
LossPresent = torch.FloatTensor
HiddenStatesPresent = List[torch.FloatTensor]
AttentionsPresent = List[torch.FloatTensor]

class HFTransformerInput(TransformersKwargs):
    labels: torch.LongTensor | None

@dataclass(frozen=True)
class HFTransformerOutput[LossT: LossPresent | None, HiddenStatesT: HiddenStatesPresent | None, AttentionsT: AttentionsPresent | None, RawT: RawHFOutputTypes]:
    logits: Logits
    loss: LossT
    hidden_states: HiddenStatesT
    attention: AttentionsT
    raw: RawT

@dataclass(frozen=True)
class HFTransformerExecutor[LossT: LossPresent | None, HiddenStatesT: HiddenStatesPresent | None, AttentionsT: AttentionsPresent | None]:
    """
    DO NOT CONSTRUCT DIRECTLY. Instead use `fresh_executor`.

    This class is intended to solve a type-convenience issue.

    Many huggingface models take in parameters such as `labels` that change the shape of the output
    (in `labels`'s case, "loss" is added to the output object, otherwise `loss` is `None`). However,
    typecheckers won't know what actual type of the output is, so to satisfy them, you will need to
    either do a cast or a manual `None` check every time.

    To solve this, you can create an executor using `fresh_executor`, then add outputs in a fluent
    interface. The outputs you add are guaranteed to be present, and the output is typed so that the
    type checker knows that. For example:
    ```
    def run(kwargs: HFTransformerInput):
        return hf_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs 
        )
    fresh_executor().include_hidden_states().execute(run).hidden_states # type safe!
    ```

    Note that the internals of this class can't be safely checked by a type checker, but as far as
    I'm aware, they are in fact type safe. Modify with caution! This class acts as a sort of "hub of
    trust" for type safety, so if you break the type safety here, other places might not know about it.
    """
    input: HFTransformerInput

    def execute[RawT: RawHFOutputTypes](self, f: Callable[[HFTransformerInput], RawT]) -> HFTransformerOutput[LossT, HiddenStatesT, AttentionsT, RawT]:
        raw = f(self.input)
        logits = cast(Logits, raw.logits) # I'm not totally sure when logits would be None. I'm doing this cast here to make the types nicer. If we run into type errors down the line we can correct it.
        loss = cast(LossT, raw.loss)
        hidden_states = cast(HiddenStatesT, None if raw.hidden_states is None else [*raw.hidden_states])
        attentions = cast(AttentionsT, None if raw.attentions is None else [*raw.attentions])
        return HFTransformerOutput(logits, loss, hidden_states, attentions, raw)
    
    def include_hidden_states(self) -> HFTransformerExecutor[LossT, List[torch.FloatTensor], AttentionsT]:
        return HFTransformerExecutor(input={
            **self.input,
            'output_hidden_states': True
        })
    
    def include_loss(self, labels: torch.LongTensor) -> HFTransformerExecutor[torch.FloatTensor, HiddenStatesT, AttentionsT]:
        return HFTransformerExecutor(input={
            **self.input,
            'labels': labels
        })
    
    def include_attentions(self) -> HFTransformerExecutor[LossT, HiddenStatesT, List[torch.FloatTensor]]:
        return HFTransformerExecutor(input={
            **self.input,
            'output_attentions': True
        })
    
    


def fresh_executor() -> HFTransformerExecutor[None, None, None]:
    return HFTransformerExecutor[None, None, None]({ **TransformersKwargs(), 'labels': None })
