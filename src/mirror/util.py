import math
import os
from pathlib import Path
import socket
import torch 

mirror_data_path = Path(
    f'/home/{os.environ['USER']}/nobackup/autodelete/mirror_data'
)

def is_login_node() -> bool:
    return 'login' in socket.gethostname()

def safe_training_run_path(training_run_id: str) -> Path:
    safe_id = training_run_id.replace(":", "-")
    return Path(f'/home/{os.environ['USER']}/nobackup/autodelete/mirror_data/training_runs') / safe_id

device = 'cpu' if is_login_node() else 'cuda'


def assert_can_download(item_name_to_download: str):
    if not is_login_node():
        raise Exception(f'Cannot download {item_name_to_download}. Try again on a login node.')

def is_power_of_ten(n: int):
    return n > 0 and math.log10(n).is_integer()
