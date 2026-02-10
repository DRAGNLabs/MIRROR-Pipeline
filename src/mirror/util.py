import math
import os
import torch 
from pathlib import Path

from mirror.config import RuntimeEnvironment, get_config
from mirror.types import TokenTensor, TokenBatch, AttentionMaskBatch

mirror_data_path = Path(f"/home/{os.environ['USER']}/nobackup/autodelete/mirror_data")

def is_login_node() -> bool:
    return get_config()['environment'] == RuntimeEnvironment.SLURM_LOGIN

def safe_training_run_path(training_run_id: str) -> Path:
    safe_id = training_run_id.replace(":", "-")
    return (mirror_data_path / "training_runs" / safe_id)

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