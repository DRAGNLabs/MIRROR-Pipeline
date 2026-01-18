import os
import socket
from typing import Literal, TypedDict

import torch

DeviceType = Literal['cpu', 'cuda']


class RuntimeConfig(TypedDict):
    device: DeviceType
    on_slurm: bool
    is_login_node: bool


_CONFIG: RuntimeConfig | None = None


def init_config(device: DeviceType | None = None) -> RuntimeConfig:
    global _CONFIG
    on_slurm = os.getenv('SLURM_JOB_ID') is not None or os.getenv('SLURM_ARRAY_JOB_ID') is not None
    is_login = on_slurm and 'login' in socket.gethostname()
    if device is None:
        if on_slurm and not is_login and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        raise ValueError(
            "Device 'cuda' was requested but CUDA is not available. "
            "Use --device cpu, or run on a CUDA-capable system."
        )
    _CONFIG = {'device': device, 'on_slurm': on_slurm, 'is_login_node': is_login}
    return _CONFIG


def get_config() -> RuntimeConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = init_config()
    return _CONFIG
