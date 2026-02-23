import torch

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.util import get_device, pad_to_longest
from mirror.row_types import TextRow
from mirror.types import TokenTensor, TokenBatch, AttentionMaskBatch

class PlaceholderPreprocessor(MirrorPreprocessor):
    @property
    def tokenization_id(self):
        return "placeholder"

    def encode(self, text):
        return torch.tensor([1, 2, 3, 4], device=get_device())

    def preprocess_example(self, example: TextRow) -> TokenTensor:
        return self.encode(example['text'])
    
    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return pad_to_longest(examples, pad_token=self.pad_token_id)

    @property
    def pad_token_id(self):
        return -1
