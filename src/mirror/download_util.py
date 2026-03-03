import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import get_token, login

from mirror.config import RuntimeEnvironment, get_config

mirror_data_path = Path(f"/home/{os.environ['USER']}/nobackup/autodelete/mirror_data")


def assert_can_download(item_name_to_download: str, *, require_hf_login: bool = False):
    config = get_config()
    if config['environment'] == RuntimeEnvironment.SLURM_COMPUTE:
        raise Exception(f'Cannot download {item_name_to_download}. Try again on a login node.')
    if require_hf_login and config["environment"] == RuntimeEnvironment.SLURM_LOGIN:
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
