
from dataclasses import asdict
import math
from pathlib import Path
import shlex
import subprocess
import sys

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from mirror.checkpoint_identifier import CheckpointIdentifier
from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.datasets.on_demand_preprocessed_dataset import OnDemandPreprocessedDataset
from mirror.models.mirror_model import MirrorModel
from mirror.models.model_util import IGNORE_ID
from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.slurm_util import SlurmConfig
from mirror.predictor import Predictor
from mirror.trainer import Trainer
from mirror.util import get_device, is_login_node


def fit(
        data: MirrorDataset,
        model: MirrorModel,
        trainer: Trainer,
        preprocessor: MirrorPreprocessor | None = None,
        checkpoint: CheckpointIdentifier | None = None,
        slurm: SlurmConfig = SlurmConfig(),
        epochs: int = 1,
        batch_size: int = 1,
        do_preprocess: bool = False,
        run_config_yaml: str = '',
        val_data: MirrorDataset | None = None,
        test_data: MirrorDataset | None = None,
        val_check_interval: int = 1,
):
    if slurm.job_type == "compute" and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=trainer.num_nodes,
            devices=trainer.devices,
        )
        print(f"Submitted batch job {job_id}")
        return

    trainer.fit(
        model,
        data,
        preprocessor,
        checkpoint,
        epochs,
        batch_size,
        do_preprocess,
        run_config_yaml,
        val_data,
        test_data,
        val_check_interval,
    )

def preprocess(
        data: MirrorDataset, 
        preprocessor: MirrorPreprocessor, 
        slurm: SlurmConfig = SlurmConfig()
) -> None:
    n_nodes = 1
    if slurm.job_type == "compute" and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=slurm.nodes or 1,
            devices=slurm.ntasks_per_node or 1,
        )
        print(f"Submitted batch job {job_id}")
        return
    
    preprocessed = data.preprocess(preprocessor.preprocess_example, n_nodes)

    total_tokens = 0
    for item in preprocessed:
        total_tokens += item.numel()
    
    print("total_tokens:", total_tokens)

def infer(
        model: MirrorModel,
        checkpoint_path: Path,
        text: str | None,
        num_tokens: int,
        preprocessor: MirrorPreprocessor | None = None,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        slurm: SlurmConfig = SlurmConfig()
) -> None:

    if text is None:
        text = input("Enter prompt: ")
        sys.argv.extend(["--text", text])

    if slurm.job_type == "compute" and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=slurm.nodes or 1,
            devices=slurm.ntasks_per_node or 1,
        )
        print(f"Submitted batch job {job_id}")
        return

    print("Beginning inference...")
    result = Predictor().predict(
        model, checkpoint_path, text, num_tokens, preprocessor,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    print(result)


def bits_per_char(
        model: MirrorModel,
        data: MirrorDataset,
        checkpoint_path: Path | None = None,
        preprocessor: MirrorPreprocessor | None = None,
        batch_size: int = 1,
        slurm: SlurmConfig = SlurmConfig(),
) -> None:
    """Evaluate bits per character on a text dataset.

    BPC = total_nats / (total_chars * ln(2)), where total_nats is the sum of
    per-token cross-entropy loss (in nats) over all non-padding predicted tokens.
    """
    if slurm.job_type == "compute" and is_login_node():
        job_id = _submit_slurm_job(
            python_args=sys.argv[1:],
            slurm=slurm,
            num_nodes=slurm.nodes or 1,
            devices=slurm.ntasks_per_node or 1,
        )
        print(f"Submitted batch job {job_id}")
        return

    if not isinstance(model, HFWhiteboxTransformer):
        raise ValueError(
            f"bits_per_char requires a model that implements HFWhiteboxTransformer, got {type(model)}"
        )

    preprocessor = preprocessor or model.preprocessor

    device = get_device()
    if checkpoint_path is not None:
        model_state = model.state_dict()
        dcp.load(
            state_dict={'model': model_state},
            checkpoint_id=str(checkpoint_path),
            no_dist=True,
        )
        model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    preprocessed = OnDemandPreprocessedDataset(data, preprocessor.preprocess_example)
    dataloader = DataLoader(
        preprocessed,
        batch_size=batch_size,
        collate_fn=preprocessor.collate,
        drop_last=False,
    )

    eos_id = preprocessor._tokenizer.eos_token_id  # type: ignore[union-attr]
    total_nats = 0.0
    total_chars = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Compute logits directly so we can get per-token losses.
            # HF shifts internally: shift_logits[i] predicts shift_ids[i+1].
            hf_out = model.hf_model(input_ids, attention_mask=attention_mask)
            shift_logits = hf_out.logits[:, :-1, :].contiguous()
            shift_ids    = input_ids[:, 1:].contiguous()
            shift_mask   = attention_mask[:, 1:].contiguous()

            # Only count text-token predictions. EOS is excluded: its cost is a
            # model artifact (knowing when a verse ends), not character prediction.
            if eos_id is not None:
                text_mask = shift_mask.bool() & (shift_ids != eos_id)
            else:
                text_mask = shift_mask.bool()

            labels_flat = shift_ids.masked_fill(~text_mask, IGNORE_ID).reshape(-1)
            per_token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                labels_flat,
                ignore_index=IGNORE_ID,
                reduction='none',
            )
            total_nats += per_token_nll.sum().item()
            total_tokens += int(text_mask.sum().item())

            # Decode non-padding tokens to count source characters (skip_special_tokens
            # strips BOS/EOS, giving the raw text character count).
            for seq, mask in zip(input_ids, attention_mask):
                non_pad_ids = seq[mask.bool()].tolist()
                text = preprocessor._tokenizer.decode(non_pad_ids, skip_special_tokens=True)  # type: ignore[union-attr]
                total_chars += len(text)

    if total_chars == 0:
        print("No characters found in dataset.")
        return

    result = total_nats / (total_chars * math.log(2))
    print(f"Bits per character: {result:.4f}")
    print(f"  Total nats:       {total_nats:.2f}")
    print(f"  Total characters: {total_chars}")
    print(f"  Total tokens:     {total_tokens}")


def _submit_slurm_job(
        *, 
        python_args: list[str], 
        slurm: SlurmConfig, 
        num_nodes: int,
        devices: int
) -> str:
    # Prevent recursion: job run should not submit again
    args = [a for a in python_args if not a.startswith("--slurm.submit")]
    args.append("--slurm.submit=false")

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("slurm.jinja")

    slurm_ctx = asdict(slurm)

    if slurm_ctx["nodes"] is None:
        slurm_ctx["nodes"] = num_nodes

    if slurm_ctx["ntasks_per_node"] is None:
        slurm_ctx["ntasks_per_node"] = devices
        
    if slurm_ctx["gpus_per_node"] is None:
        slurm_ctx["gpus_per_node"] = devices

    context = {
        **slurm_ctx,
        "chdir": str(Path.cwd()),
        "activate_cmd": "mamba activate ./.env",
        "run_cmd": f"srun python {sys.argv[0]} {shlex.join(python_args)}",
    }

    script = template.render(**context)
    
    res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {res.returncode}):\n{res.stderr}\n\nGenerated script:\n{script}"
        )
    return res.stdout.strip().split()[-1]
