import os
import shutil
from typing import cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mirror.download_util import assert_can_download
from mirror.types import AttentionMaskBatch, IGNORE_ID, LabeledTokens, LabelsBatch, TokenBatch
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
) -> tuple[TokenBatch, AttentionMaskBatch, LabelsBatch]:
    device = get_device()
    batch = tokenizer.pad(
        {"input_ids": [e["input_ids"] for e in examples]},
        padding=True,
        return_tensors="pt",
    ).to(device)
    input_ids = cast(TokenBatch, batch["input_ids"])
    attention_mask = cast(AttentionMaskBatch, batch["attention_mask"])

    padded_len = input_ids.shape[1]
    pad_left = getattr(tokenizer, "padding_side", "right") == "left"
    labels = torch.full((len(examples), padded_len), IGNORE_ID, dtype=torch.long, device=device)
    for i, e in enumerate(examples):
        row = torch.as_tensor(e["labels"], dtype=torch.long, device=device)
        if pad_left:
            labels[i, padded_len - row.shape[0]:] = row
        else:
            labels[i, :row.shape[0]] = row
    return input_ids, attention_mask, cast(LabelsBatch, labels)
