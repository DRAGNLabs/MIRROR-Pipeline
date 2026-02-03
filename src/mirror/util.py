import math
import os
from dotenv import load_dotenv
from huggingface_hub import get_token, login
from pathlib import Path
import sys
import torch 
from mirror.types import TokenTensor, TokenBatch, AttentionMaskBatch
from mirror.config import RuntimeEnvironment, get_config

mirror_data_path = Path(f"/home/{os.environ['USER']}/nobackup/autodelete/mirror_data")

def is_login_node() -> bool:
    return get_config()['environment'] == RuntimeEnvironment.SLURM_LOGIN

def safe_training_run_path(training_run_id: str) -> Path:
    safe_id = training_run_id.replace(":", "-")
    return (Path(f"/home/{os.environ['USER']}/nobackup/autodelete/mirror_data/training_runs") / safe_id)

def get_device() -> str:
    return get_config()['device']

def assert_can_download(item_name_to_download: str, *, require_hf_login: bool = False):
    config = get_config()
    if config['environment'] == RuntimeEnvironment.SLURM_COMPUTE:
        raise Exception(f'Cannot download {item_name_to_download}. Try again on a login node.')
    if require_hf_login:
        if config["environment"] == RuntimeEnvironment.SLURM_LOGIN:
            load_dotenv(".ENV", override=False)
            token = os.getenv("HUGGINGFACE_HUB_TOKEN") or get_token()
            if token:
                return
            login()
            token = os.getenv("HUGGINGFACE_HUB_TOKEN") or get_token()
            if not token:
                raise RuntimeError(
                    "Hugging Face login did not produce a cached token. "
                    "Try `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN` in `.ENV`."
                )

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

def normalize_model_argv(argv: list[str]) -> list[str]:
    model_modules = {
        "MirrorLlamaModel": "mirror_llama_model",
        "MirrorGPTModel": "mirror_gpt_model"
    }
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--model.class_path"):
            i += 2 if a == "--model.class_path" else 1
            continue

        name = None
        if a in ("--model", "--model.name"):
            name = argv[i + 1]
            i += 2
        elif a.startswith("--model=") or a.startswith("--model.name="):
            name = a.split("=", 1)[1]
            i += 1
        if name is not None:
            if "." not in name:
                mod = model_modules.get(name)
                name = f"mirror.models.{mod}.{name}"
            out.extend(["--model.class_path", name])
            continue

        if a.startswith("--model.") and not a.startswith(
            ("--model.init_args.", "--model.class_path", "--model.name", "--model.help")
        ):
            out.append("--model.init_args." + a[len("--model."):])
            i += 1
            continue

        out.append(a)
        i += 1
    return out
