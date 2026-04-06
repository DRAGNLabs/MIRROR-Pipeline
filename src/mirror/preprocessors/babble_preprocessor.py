import os
from typing import cast

from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import PreTrainedTokenizerFast

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import collate_tokens
from mirror.types import TokenTensor, TokenBatch, AttentionMaskBatch
from mirror.row_types import TextRow

from mirror.util import mirror_data_path

class BabblePreprocessor(
    MirrorPreprocessor[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch]]
):
    def __init__(self, file_path: str, vocab_size: int) -> None:
        tokenizer_path = f"{mirror_data_path}/tokenizers/babble_tokenizer/"
        os.makedirs(tokenizer_path, exist_ok=True)

        tokenizer_file = tokenizer_path + f"tokenizer-{vocab_size}.json"

        if os.path.exists(tokenizer_file):
            raw_tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            raw_tokenizer = ByteLevelBPETokenizer()
            raw_tokenizer.train(
                files=[file_path],
                vocab_size=vocab_size,
                min_frequency=5,
                special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
            )
            raw_tokenizer.save(tokenizer_file)

        raw_tokenizer.enable_padding(pad_token="<pad>", pad_id=raw_tokenizer.token_to_id("<pad>"))
        self._raw_tokenizer = raw_tokenizer

        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

    def preprocess_example(self, example: TextRow) -> TokenTensor:
        encoding = self._raw_tokenizer.encode(example['text'], add_special_tokens=True)
        ids = encoding.ids
        if len(ids) < 2:
            eos = self._raw_tokenizer.token_to_id("</s>")
            ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
        return cast(TokenTensor, ids)

    def collate(self, examples: list[TokenTensor]) -> tuple[TokenBatch, AttentionMaskBatch]:
        return collate_tokens(self._tokenizer, examples)

    @property
    def pad_token_id(self) -> int:
        return int(self._tokenizer.pad_token_id)  # type: ignore[arg-type]
