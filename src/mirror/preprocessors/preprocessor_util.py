import os
import shutil
from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mirror.download_util import assert_can_download
from mirror.types import AttentionMaskBatch, IGNORE_ID, LabeledTokens, LabelsBatch, StandardBatch, TokenBatch
from mirror.util import get_device, mirror_data_path

tokenizers_path = mirror_data_path / "tokenizers"


def load_hf_tokenizer(
        hf_model_name: str,
        reset_cache: bool = False,
) -> PreTrainedTokenizerBase:
    """
    Cache tokenizer artifacts under mirror_data/tokenizers/<hf_model_name>.
    """
    tokenizer_path = tokenizers_path / hf_model_name
    tokenizer_id = hf_model_name
    hf_cache_path = mirror_data_path / "hf_cache"

    if reset_cache:
        shutil.rmtree(tokenizer_path, ignore_errors=True)

    if os.path.exists(tokenizer_path):  # cached
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    else:
        assert_can_download(tokenizer_id, require_hf_login=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=hf_cache_path)
        tokenizer.save_pretrained(tokenizer_path)

    return tokenizer


def collate_tokens(
    tokenizer: PreTrainedTokenizerBase,
    examples: list[LabeledTokens],
) -> StandardBatch:
    device = get_device()
    batch = tokenizer.pad(
        {"input_ids": [e["input_ids"] for e in examples]},
        padding=True,
        return_tensors="pt",
    ).to(device)
    input_ids = cast(TokenBatch, batch["input_ids"])
    attention_mask = cast(AttentionMaskBatch, batch["attention_mask"])

    # Pad labels via the same tokenizer.pad call so padding direction/length stay
    # in sync with input_ids, then overwrite padded positions with IGNORE_ID.
    padded_labels = cast(LabelsBatch, tokenizer.pad(
        {"input_ids": [e["labels"] for e in examples]},
        padding=True,
        return_tensors="pt",
    ).to(device)["input_ids"])
    labels = padded_labels.masked_fill(attention_mask == 0, IGNORE_ID)
    return input_ids, attention_mask, cast(LabelsBatch, labels)
