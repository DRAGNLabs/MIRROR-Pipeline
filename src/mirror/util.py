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


device = 'cpu' if is_login_node() else 'cuda'


def assert_can_download(item_name_to_download: str):
    if not is_login_node():
        raise Exception(f'Cannot download {item_name_to_download}. Try again on a login node.')

def is_power_of_ten(n: int):
    return n > 0 and math.log10(n).is_integer()

def pad_to_longest(batch, pad_token):
    dev = torch.device(device)
    lens = torch.tensor([b.numel() for b in batch], device=dev, dtype=torch.long)
    max_len = lens.max().item()
    batch_size = len(batch)

    tokens = torch.full((batch_size, max_len), int(pad_token), dtype=torch.long, device=dev)

    ar = torch.arange(max_len, device=dev)
    attention_mask = (ar.unsqueeze(0) < lens.unsqueeze(1)).to(torch.long)

    print("raw batch[0]:", batch[0])
    print("lens:", lens)
    print("attention_mask:\n", attention_mask)

    for i, b in enumerate(batch):
        L = b.numel()
        tokens[i,:L] = b
    
    print("padded tokens[0]: ", tokens[0])
    
    return tokens, attention_mask