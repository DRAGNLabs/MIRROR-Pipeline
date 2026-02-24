from dataclasses import asdict
import functools
from typing import Any, Callable, Concatenate, List, Protocol, Tuple, TypedDict, cast

from attr import dataclass
import torch
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast

from mirror.models.mirror_gpt_model import MirrorGPTModel
from mirror.models.mirror_llama_model import MirrorLlamaModel

class CoreDict(TypedDict): pass

class Basic[InT: CoreDict, OutT](Protocol):
    def __call__(self, args: InT) -> OutT: ...


class LabelsDict(CoreDict):
    labels: torch.LongTensor

class TakesLabels[InT: LabelsDict, OutT](Protocol):
    def __call__(self, args: InT) -> OutT: ...

def tuplize[**P, OutT](f: Callable[P, OutT]) -> Callable[P, Tuple[OutT]]:
    def tuplized(*args: P.args, **kwargs: P.kwargs) -> Tuple[OutT]:
        return f(*args, **kwargs),
    return tuplized

def with_loss[**P, *OutT, HFModelOutputT: CausalLMOutputWithPast | CausalLMOutputWithCrossAttentions](callable: Callable[P, Tuple[*OutT, HFModelOutputT]]) -> Callable[P, Tuple[torch.FloatTensor, *OutT, HFModelOutputT]]:
    def hf_callabale_with_loss(*args: P.args, **kwargs: P.kwargs) -> Tuple[torch.FloatTensor, *OutT, HFModelOutputT]: 
        main_out, hf_out = callable(*args, **kwargs)
        if hf_out.loss is None:
            raise Exception()

        return hf_out.loss, main_out, hf_out
    return hf_callabale_with_loss

test = functools.partial(MirrorGPTModel().hf_model.forward, labels=torch.LongTensor([42]))
