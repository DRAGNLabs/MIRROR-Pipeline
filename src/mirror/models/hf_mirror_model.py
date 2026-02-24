from abc import ABC

from transformers import PreTrainedModel

from mirror.models.mirror_model import MirrorModel


class HFMirrorModel[RawT, ProcessedT, BatchT, HFModelT: PreTrainedModel](MirrorModel[RawT, ProcessedT, BatchT], ABC):
    hf_model: HFModelT