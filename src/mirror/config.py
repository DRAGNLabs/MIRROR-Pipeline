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


def _is_on_slurm() -> bool:
    return os.getenv('SLURM_JOB_ID') is not None or os.getenv('SLURM_ARRAY_JOB_ID') is not None


def _is_login_node() -> bool:
    return 'login' in socket.gethostname()


def _resolve_device(requested: DeviceType | None, is_login_node: bool) -> DeviceType:
    if requested is not None:
        return requested
    if is_login_node:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def init_config(device: DeviceType | None = None) -> RuntimeConfig:
    global _CONFIG
    on_slurm = _is_on_slurm()
    is_login = _is_login_node()
    resolved_device = _resolve_device(device, is_login)
    _CONFIG = {'device': resolved_device, 'on_slurm': on_slurm, 'is_login_node': is_login}
    return _CONFIG


def get_config() -> RuntimeConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = init_config()
    return _CONFIG
