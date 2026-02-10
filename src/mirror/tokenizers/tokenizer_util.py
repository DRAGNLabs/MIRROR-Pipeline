import os
import shutil

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mirror.download_util import assert_can_download, mirror_data_path

tokenizers_path = mirror_data_path / "tokenizers"


def load_hf_tokenizer[HfTokenizerT: PreTrainedTokenizerBase](
        hf_model_name: str | None = None,
        reset_cache: bool = False,
) -> HfTokenizerT:
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
