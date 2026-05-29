from abc import ABC, abstractmethod

import torch.nn as nn
from transformers import PreTrainedModel

from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.preprocessors.mirror_preprocessor import InferenceFriendlyPreprocessor


class InferenceFriendlyModel(nn.Module, HFWhiteboxTransformer, ABC):
    @property
    @abstractmethod
    def hf_model(self) -> PreTrainedModel: ...

    @property
    @abstractmethod
    def preprocessor(self) -> InferenceFriendlyPreprocessor: ...
