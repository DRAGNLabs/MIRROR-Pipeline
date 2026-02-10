import os
import shutil
import importlib
from typing import Type

from lightning import Fabric
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig

from mirror.download_util import assert_can_download, mirror_data_path
from mirror.models.mirror_model import MirrorModel


def load_hf_model_from_cache_or_download(
        hf_model_name: str | None = None,
        reset_cache: bool = False,
        model_cls: Type[PreTrainedModel] = AutoModel,
) -> PreTrainedModel:
    """
    The first time this is called with a particular model, it will download
    the model from huggingface and cache it under mirror_data/models/<hf_model_name>.
    Thereafter, if it is called again with reset_cache=False, it will use the cached
    data from the first run.
    """
    models_path = mirror_data_path / "models"
    hf_cache_path = mirror_data_path / "hf_cache"
    model_path = models_path / hf_model_name
    model_id = hf_model_name

    if reset_cache:
        shutil.rmtree(model_path, ignore_errors=True)

    if model_path.exists():
        model = model_cls.from_pretrained(model_path, local_files_only=True)
        return model

    assert_can_download(model_id, require_hf_login=True)
    model = model_cls.from_pretrained(model_id, cache_dir=hf_cache_path)
    model.save_pretrained(model_path)

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
    models_path = mirror_data_path / "models"
    model_path = models_path / hf_model_name

    if reset_cache:
        os.remove(model_path / "config.json")

    if model_path.exists():  # cached
        return AutoConfig.from_pretrained(model_path, local_files_only=True)

    # Avoid creating config-only directory 
    load_hf_model_from_cache_or_download(hf_model_name=hf_model_name, reset_cache=reset_cache)

    return AutoConfig.from_pretrained(model_path, local_files_only=True)

def build_causal_lm(
        model_name: str | None = None,
        weights: str = "pretrained"
) -> MirrorModel:
    match weights:
        case "pretrained":
            model = load_hf_model_from_cache_or_download(
                model_name,
                model_cls=AutoModelForCausalLM,
            )
        case "random":
            config = load_hf_config_from_cache_or_download(model_name)
            model = AutoModelForCausalLM.from_config(config)
    return model


def instantiate_model(model: object, *, fabric: Fabric) -> MirrorModel:
    if isinstance(model, MirrorModel):
        return model
    with fabric.init_module():
        class_path = model.class_path
        init_args = getattr(model, "init_args", None)
        if init_args is None:
            kwargs = {}
        elif isinstance(init_args, dict):
            kwargs = init_args
        else:
            kwargs = vars(init_args)
        module_name, _, class_name = class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**kwargs)
