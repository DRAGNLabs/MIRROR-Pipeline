import os
import shutil
from typing import Type

from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig

from mirror.config import RuntimeEnvironment, get_config
from mirror.util import assert_can_download, mirror_data_path


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
    models_path = mirror_data_path / 'models'
    hf_cache_path = mirror_data_path / "hf_cache"
    model_path = models_path / hf_model_name
    model_id = hf_model_name

    if reset_cache:
        shutil.rmtree(model_path, ignore_errors=True)

    path_exists = model_path.exists()
    weights_present = path_exists and (
        any(
            (model_path / name).exists()
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )
        or any(model_path.glob("model-*.safetensors"))
        or any(model_path.glob("pytorch_model-*.bin"))
    )

    if weights_present:  # cached
        model = model_cls.from_pretrained(model_path, local_files_only=True)
    else:
        if path_exists:
            # Config-only cache created for random init.
            shutil.rmtree(model_path, ignore_errors=True)
        assert_can_download(model_id, require_hf_login=True)
        # Keep HF's internal cache layout (models--org--repo, blobs/, refs/, etc.)
        # out of our `mirror_data/models/...` tree.
        model = model_cls.from_pretrained(model_id, cache_dir=hf_cache_path)
        model.save_pretrained(model_path)

    # Some HF configs ship `loss_type=None`, which triggers a warning in newer
    # `transformers`. For CausalLM models we always want the default.
    if hasattr(model, "config") and getattr(model.config, "loss_type", "MISSING") is None:
        model.config.loss_type = "ForCausalLMLoss"

    return model

def load_hf_config_from_cache_or_download(
	        hf_model_name: str | None = None,
	        reset_cache: bool = False,
) -> PretrainedConfig:
    """
    Like load_hf_model_from_cache_or_download, but only ensures the config is
    available locally and returns a config object. Useful for random-weight
    initialization via `model_cls.from_config(...)`.
    """
    model_configs_path = mirror_data_path / "model_configs"
    hf_cache_path = mirror_data_path / "hf_cache"
    model_path = model_configs_path / hf_model_name
    model_id = hf_model_name

    if reset_cache:
        shutil.rmtree(model_path, ignore_errors=True)

    if os.path.exists(model_path):  # cached
        return AutoConfig.from_pretrained(model_path, local_files_only=True)

    if get_config()["environment"] == RuntimeEnvironment.SLURM_COMPUTE:
        raise RuntimeError(
            f"Model config for '{model_id}' is not cached locally, and downloads are disabled on compute nodes.\n"
            "Try again on a login node, or use `--model.weights=pretrained`."
        )

    assert_can_download(model_id, require_hf_login=True)
    # Keep HF cache layout out of our `mirror_data/model_configs/...` tree.
    config = AutoConfig.from_pretrained(model_id, cache_dir=hf_cache_path)
    model_path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(model_path)
    return config
