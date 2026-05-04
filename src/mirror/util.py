import math
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch.nn as nn

from datasets import config as datasets_config
from mirror.config import RuntimeEnvironment, get_config


_CONFIGS_DIR = Path(__file__).parent.parent.parent / 'configs'

def resolve_config_args(args: list[str]) -> list[str]:
    result = list(args)
    for i, arg in enumerate(result[:-1]):
        if arg == '--config':
            path = Path(result[i + 1])
            if not path.exists() and (_CONFIGS_DIR / path).exists():
                result[i + 1] = str(_CONFIGS_DIR / path)
    return result

mirror_data_path = Path(
    os.getenv("MIRROR_DATA_PATH", f"/home/{os.environ['USER']}/nobackup/autodelete/mirror_data")
)

def is_login_node() -> bool:
    return get_config()['environment'] == RuntimeEnvironment.SLURM_LOGIN

def is_compute_node() -> bool:
    return get_config()['environment'] == RuntimeEnvironment.SLURM_COMPUTE

def safe_training_run_path(training_run_id: str) -> Path:
    safe_id = training_run_id.replace(":", "-")
    return (mirror_data_path / "training_runs" / safe_id)

def get_device() -> str:
    return get_config()['device']

def is_power_of_ten(n: int):
    return n > 0 and math.log10(n).is_integer()

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@contextmanager
def _ds_cache_path_context() -> Generator[None, None, None]:
    hf_cache_path = mirror_data_path / "hf_cache"
    hf_cache_path.mkdir(exist_ok=True)
    original = datasets_config.HF_DATASETS_CACHE
    datasets_config.HF_DATASETS_CACHE = str(hf_cache_path)
    try:
        yield
    finally:
        datasets_config.HF_DATASETS_CACHE = original
    