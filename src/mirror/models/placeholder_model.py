import torch
import torch.optim as optim
import torch.nn as nn

from mirror.models.mirror_model import MirrorModel
from mirror.tokenizers.placeholder_tokenizer import PlaceholderTokenizer
from mirror.types import AttentionMaskBatch, Loss, TokenBatch, TokenTensor
from mirror.util import get_device, pad_to_longest


class PlaceholderModel(MirrorModel[str, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([0.0], device=get_device()))
        self._tokenizer = PlaceholderTokenizer()

    @property
    def tokenizer(self) -> PlaceholderTokenizer:
        return self._tokenizer

    def preprocess_example(self, text: str) -> TokenTensor:
        return torch.tensor(self._tokenizer.encode(text))

    def get_column_names(self, data_column: str, label_column: str):
        self.data_column = data_column
        self.label_column = label_column

    def preprocess_row[RawT](self, row: dict[str,RawT]) -> TokenTensor:
        text = row[self.data_column] # This placeholder doesn't use labels so it's left out
        label = row[self.label_column]
        return {'input_ids': torch.tensor(self._tokenizer.encode(text))}

    def get_preprocess_row_function(self, dataset: MirrorDataset):
        dataset.data_column

    
    def training_step(self, batch: tuple[TokenBatch, AttentionMaskBatch]) -> Loss:
        tokens, attention_mask = batch
        return self.parameter

    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return pad_to_longest(examples, pad_token=self.tokenizer.pad_token_id)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters())
