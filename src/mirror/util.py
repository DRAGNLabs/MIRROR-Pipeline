import math
import os
from pathlib import Path
import socket
import torch 
from mirror.types import TokenTensor

mirror_data_path = Path(
    f'/home/{os.environ['USER']}/nobackup/autodelete/mirror_data'
)


def is_login_node() -> bool:
    return 'login' in socket.gethostname()


device = 'cpu' if is_login_node() else 'cuda'


def assert_can_download(item_name_to_download: str):
    if not is_login_node():
        raise Exception(f'Cannot download {item_name_to_download}. Try again on a login node.')

def is_power_of_ten(n: int):
    return n > 0 and math.log10(n).is_integer()