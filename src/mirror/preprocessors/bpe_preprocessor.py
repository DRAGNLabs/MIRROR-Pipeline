import hashlib
import os
from pathlib import Path
from typing import cast

from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import PreTrainedTokenizerFast
from typed_datasets import TypedDataset

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.preprocessors.infer_friendly_preprocessor import InferFriendlyPreprocessor
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import collate_tokens
from mirror.types import LabeledTokens, StandardBatch, TextRow

from mirror.util import _ds_cache_path_context, mirror_data_path

_SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<|user|>", "<|assistant|>"]

_CHAT_TEMPLATE = (
    "<s>"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|user|>{{ message['content'] }}</s>"
    "{% elif message['role'] == 'assistant' %}"
    "{% generation %}<|assistant|>{{ message['content'] }}</s>{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
)


class BPEPreprocessor(
    InferFriendlyPreprocessor,
    MirrorPreprocessor[TextRow, LabeledTokens, StandardBatch],
):
    def __init__(self, file_path: Path, vocab_size: int) -> None:
        file_hash = hashlib.md5(f"{str(file_path)}_v1.2".encode()).hexdigest()[:8]
        tokens_hash = hashlib.md5(str(_SPECIAL_TOKENS).encode()).hexdigest()[:4]
        tokenizer_path = f"{mirror_data_path}/tokenizers/bpe_{file_hash}_{vocab_size}_{tokens_hash}/"
        os.makedirs(tokenizer_path, exist_ok=True)

        tokenizer_file = tokenizer_path + "tokenizer.json"

        if os.path.exists(tokenizer_file):
            raw_tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            raw_tokenizer = ByteLevelBPETokenizer()
            raw_tokenizer.train(
                files=[str(file_path)],
                vocab_size=vocab_size,
                min_frequency=5,
                special_tokens=_SPECIAL_TOKENS,
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
            additional_special_tokens=["<|user|>", "<|assistant|>"],
        )
        self._tokenizer.chat_template = _CHAT_TEMPLATE

    def format_data(self, data_source: MirrorDataset[TextRow]) -> TypedDataset[LabeledTokens]:
        raw_tokenizer = self._raw_tokenizer

        def tokenize(row: TextRow) -> LabeledTokens:
            encoding = raw_tokenizer.encode(row['text'], add_special_tokens=True)
            ids = encoding.ids
            if len(ids) < 2:
                eos = raw_tokenizer.token_to_id("</s>")
                ids = [eos, eos] if len(ids) == 0 else [*ids, eos]
            return LabeledTokens(input_ids=ids, labels=list(ids))

        with _ds_cache_path_context():
            return data_source.ds.map(tokenize, remove_columns=list(data_source.ds.columns))

    def collate(self, examples: list[LabeledTokens]) -> StandardBatch:
        return collate_tokens(self._tokenizer, examples)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    @property
    def pad_token_id(self) -> int:
        return int(cast(int, self._tokenizer.pad_token_id))
