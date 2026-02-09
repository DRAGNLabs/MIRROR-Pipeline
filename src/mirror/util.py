import math
import os
import torch 
from lightning import Fabric
from pathlib import Path

from mirror.config import RuntimeEnvironment, get_config
from mirror.download_util import assert_can_download, mirror_data_path
from mirror.models.mirror_model import MirrorModel
from mirror.types import TokenTensor, TokenBatch, AttentionMaskBatch
import importlib

def is_login_node() -> bool:
    return get_config()['environment'] == RuntimeEnvironment.SLURM_LOGIN

def safe_training_run_path(training_run_id: str) -> Path:
    safe_id = training_run_id.replace(":", "-")
    return (Path(f"/home/{os.environ['USER']}/nobackup/autodelete/mirror_data/training_runs") / safe_id)

def get_device() -> str:
    return get_config()['device']

def is_power_of_ten(n: int):
    return n > 0 and math.log10(n).is_integer()

def pad_to_longest(batch: list[TokenTensor], pad_token: int) -> tuple[TokenBatch, AttentionMaskBatch]:
    device = get_device()
    
    lens = torch.tensor([b.numel() for b in batch], device=device, dtype=torch.long)
    max_len = lens.max().item()
    batch_size = len(batch)

    tokens = torch.full((batch_size, max_len), int(pad_token), dtype=torch.long, device=device)

    ar = torch.arange(max_len, device=device)
    attention_mask = (ar.unsqueeze(0) < lens.unsqueeze(1)).to(torch.long)

    for i, b in enumerate(batch):
        L = b.numel()
        tokens[i,:L] = b
    
    return tokens, attention_mask

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
