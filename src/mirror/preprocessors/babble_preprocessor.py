import os
from typing import cast

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import collate_tokens
from mirror.types import TokenTensor, TokenBatch, AttentionMaskBatch
from mirror.row_types import TextRow

from mirror.util import mirror_data_path

class BabblePreprocessor(
    MirrorPreprocessor[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]
):
    def __init__(self, path_to_txt_ds: str, vocab_size: int) -> None:
        tokenizer_path = f"{mirror_data_path}/tokenizers/babble_tokenizer/"
        vocab_file = f"{tokenizer_path}/babble-{vocab_size}-vocab.json"
        merges_file = f"{tokenizer_path}/babble-{vocab_size}-merges.txt"

        if os.path.exists(vocab_file) and os.path.exists(merges_file):
            self._tokenizer = ByteLevelBPETokenizer.from_file(vocab=vocab_file, merges=merges_file)
        else:
            self._tokenizer = ByteLevelBPETokenizer()
            self._tokenizer.train(
                files=[path_to_txt_ds],
                vocab_size=vocab_size,
                min_frequency=5,
                special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
            )
            self._tokenizer.save_model(tokenizer_path, prefix=f"babble-{vocab_size}")
        
        self._tokenizer.enable_padding(pad_token="<pad>", pad_id=self._tokenizer.token_to_id("<pad>"))

        self._hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
)

    def preprocess_example(self, example: TextRow) -> TokenTensor:
        encoding = self._tokenizer.encode(example['text'], add_special_tokens=True)
        ids = encoding.ids
        if len(ids) < 2:
            eos = self._tokenizer.token_to_id("</s>")
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
        return cast(TokenTensor, ids)

    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return collate_tokens(self._hf_tokenizer, examples)

    @property
    def pad_token_id(self) -> int:
        return int(self._hf_tokenizer.pad_token_id)  # type: ignore[arg-type]
