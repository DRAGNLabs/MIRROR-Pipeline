import os
import shutil
from typing import Type

from transformers import AutoModel, PreTrainedModel

from mirror.util import assert_can_download, mirror_data_path

models_path = mirror_data_path / 'models'


def load_hf_model_from_cache_or_download(
        hf_model_name: str | None = None,
        reset_cache: bool = False,
        model_cls: Type[PreTrainedModel] = AutoModel,
) -> PreTrainedModel:
    """
    The first time this is called with a particular path/name pair, it will download
    the model from huggingface and cache it under mirror_data/models/<hf_model_name>.
    Thereafter, if it is called again with reset_cache=False, it will use the cached
    data from the first run.
    """
    model_path = models_path / hf_model_name
    model_id = hf_model_name

    if reset_cache:
        shutil.rmtree(model_path, ignore_errors=True)

    if os.path.exists(model_path): # cached
        model = model_cls.from_pretrained(model_path, local_files_only=True)
    else:
        assert_can_download(model_id)
        model = model_cls.from_pretrained(model_id, cache_dir=models_path)
        model.save_pretrained(model_path)

    return model
