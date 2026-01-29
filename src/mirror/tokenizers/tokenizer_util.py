import os
import shutil
from typing import Type

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mirror.util import assert_can_download, mirror_data_path

tokenizers_path = mirror_data_path / 'models'


def load_hf_tokenizer_from_cache_or_download(
        hf_model_name: str | None = None,
        reset_cache: bool = False,
        tokenizer_cls: Type[PreTrainedTokenizerBase] = AutoTokenizer,
) -> PreTrainedTokenizerBase:
    """
    Cache tokenizer artifacts under mirror_data/models/<hf_model_name>.
    """
    tokenizer_path = tokenizers_path / hf_model_name
    tokenizer_id = hf_model_name

    if reset_cache:
        shutil.rmtree(tokenizer_path)

    is_cached = os.path.exists(tokenizer_path)

    if is_cached:
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_path)
    else:
        assert_can_download(tokenizer_id)
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_id, cache_dir=tokenizers_path)
        tokenizer.save_pretrained(tokenizer_path)

    return tokenizer
