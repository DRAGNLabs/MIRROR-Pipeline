import os
import socket
from enum import Enum
from typing import Literal, TypedDict

import torch

DeviceType = Literal['cpu', 'cuda']


class RuntimeEnvironment(str, Enum):
    LOCAL = 'local'
    SLURM_LOGIN = 'slurm-login'
    SLURM_COMPUTE = 'slurm-compute'


class RuntimeConfig(TypedDict):
    device: DeviceType
    environment: RuntimeEnvironment


_CONFIG: RuntimeConfig | None = None


def init_config(device: DeviceType | None = None) -> RuntimeConfig:
    global _CONFIG
    
    if 'login' in socket.gethostname():
        environment = RuntimeEnvironment.SLURM_LOGIN
    elif os.getenv('SLURM_JOB_ID') is not None or os.getenv('SLURM_ARRAY_JOB_ID') is not None:
        environment = RuntimeEnvironment.SLURM_COMPUTE
    else:
        environment = RuntimeEnvironment.LOCAL
    if device is None:
        if environment == RuntimeEnvironment.SLURM_COMPUTE and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        raise ValueError(
            "Device 'cuda' was requested but CUDA is not available. "
            "Use --device cpu, or run on a CUDA-capable system."
        )
    _CONFIG = {'device': device, 'environment': environment}
    return _CONFIG


def get_config() -> RuntimeConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = init_config()
    return _CONFIG
