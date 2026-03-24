import math
import os
from pathlib import Path
import datasets
from mirror.config import RuntimeEnvironment, get_config


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

def set_ds_cache_path():
    hf_cache_path =  mirror_data_path / "hf_cache"
    hf_cache_path.mkdir(exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = hf_cache_path