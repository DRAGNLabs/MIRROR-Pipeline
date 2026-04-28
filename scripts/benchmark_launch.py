#!/usr/bin/env python
"""Grid-search benchmark launcher (runs on login node).

For each combination of device configuration, batch size, and model size,
submits an sbatch job unless the configuration is already running (lock file
exists) or already completed (entry in the JSONL log).
"""
import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from mirror.benchmark_configs import MODEL_CONFIGS
from mirror.callbacks.timer_callback import benchmark_lock_path, benchmark_run_key
from mirror.models.mirror_llama_model import MirrorLlamaModel
from mirror.util import count_params, mirror_data_path


DEVICE_CONFIGS: list[tuple[int, int]] = [
    (1, 1),
    (1, 8),
    (2, 8),
    (4, 8),
]
BATCH_SIZES: list[int] = [1, 4, 16, 64]


def _model_param_counts() -> dict[str, int]:
    return {
        label: count_params(MirrorLlamaModel(initialization=cfg))
        for label, cfg in MODEL_CONFIGS.items()
    }


def _completed_keys(log_file: Path) -> set[str]:
    if not log_file.exists():
        return set()
    completed = set()
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                completed.add(
                    benchmark_run_key(
                        entry["num_nodes"],
                        entry["devices_per_node"],
                        entry["batch_size"],
                        entry["param_count"],
                    )
                )
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def _sbatch_script(
    *,
    num_nodes: int,
    devices: int,
    batch_size: int,
    model_size: str,
    gpu_type: str,
    log_file: Path,
    lock_dir: Path,
    time_limit: str,
) -> str:
    run_cmd = (
        f"srun python {Path(__file__).parent / 'benchmark_run.py'}"
        f" --num-nodes {num_nodes}"
        f" --devices {devices}"
        f" --batch-size {batch_size}"
        f" --model-size {model_size}"
        f" --log-file {log_file}"
        f" --lock-dir {lock_dir}"
    )
    return textwrap.dedent(f"""\
        #!/bin/bash --login
        #SBATCH --time={time_limit}
        #SBATCH --nodes={num_nodes}
        #SBATCH --ntasks-per-node={devices}
        #SBATCH --gpus-per-node={gpu_type}:{devices}
        #SBATCH --mem-per-cpu=16G
        #SBATCH --qos=dw87
        #SBATCH --output={mirror_data_path}/slurm_logs/%j.out
        #SBATCH --open-mode=append
        #SBATCH --signal=SIGHUP@90
        #SBATCH --chdir={Path.cwd()}

        mamba activate ./.env

        {run_cmd}
    """)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-type",   default="h100", help="SLURM GPU type (e.g. h100, h200, a100)")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--log-file",   type=Path, default=mirror_data_path / "benchmark_log.jsonl")
    parser.add_argument("--lock-dir",   type=Path, default=mirror_data_path / "timer_locks")
    parser.add_argument("--dry-run",    action="store_true", help="Print configs without submitting")
    args = parser.parse_args()

    (mirror_data_path / "slurm_logs").mkdir(parents=True, exist_ok=True)
    args.lock_dir.mkdir(parents=True, exist_ok=True)

    param_counts = _model_param_counts()
    completed = _completed_keys(args.log_file)

    submitted = skipped_running = skipped_done = 0

    for num_nodes, devices in DEVICE_CONFIGS:
        for batch_size in BATCH_SIZES:
            for model_size, param_count in param_counts.items():
                run_key = benchmark_run_key(num_nodes, devices, batch_size, param_count)

                if run_key in completed:
                    skipped_done += 1
                    continue

                lock = benchmark_lock_path(args.lock_dir, num_nodes, devices, batch_size, param_count)
                if lock.exists():
                    print(f"[skip - running] {run_key}")
                    skipped_running += 1
                    continue

                script = _sbatch_script(
                    num_nodes=num_nodes,
                    devices=devices,
                    batch_size=batch_size,
                    model_size=model_size,
                    gpu_type=args.gpu_type,
                    log_file=args.log_file,
                    lock_dir=args.lock_dir,
                    time_limit=args.time_limit,
                )

                if args.dry_run:
                    print(f"[dry-run] would submit: {run_key}")
                    submitted += 1
                    continue

                res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
                if res.returncode != 0:
                    print(f"sbatch failed for {run_key}:\n{res.stderr}", file=sys.stderr)
                    continue
                job_id = res.stdout.strip().split()[-1]
                print(f"[submitted {job_id}] {run_key}")
                submitted += 1

    print(f"\nsubmitted={submitted}  skipped_running={skipped_running}  skipped_done={skipped_done}")


if __name__ == "__main__":
    main()
