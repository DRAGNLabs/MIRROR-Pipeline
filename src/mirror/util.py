import math
import os
from pathlib import Path

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
