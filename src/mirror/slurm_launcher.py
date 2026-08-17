import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.slurm_util import SlurmConfig, expand_grid, grid_override_args, parse_grid, parse_slurm_config
from mirror.util import is_compute_node, is_login_node


def submit_slurm_jobs(python_args: list[str]) -> None:
    """Submit one compute job per `grid` combination (one job if no grid), then exit.

    `python_args` is everything after the python script (e.g. `sys.argv[1:]`),
    starting with the subcommand. No-op when not launching a compute job from a
    login node, so the caller proceeds to run in-process. A grid that would not
    be submitted is an error rather than being silently ignored.
    """
    grid = parse_grid(python_args)
    _reject_ignored_grid(grid, python_args[0])
    if not is_login_node():
        return
    if parse_slurm_config(python_args).job_type != "compute":
        if grid:
            sys.exit("Grid search requires `slurm.job_type: compute`; change the job type or remove the `grid` section.")
        return

    for combo in expand_grid(grid):
        args = python_args + grid_override_args(combo)
        _submit_one(parse_slurm_config(args), args, combo)
    sys.exit(0)


def _reject_ignored_grid(grid: dict[str, Any], subcommand: str) -> None:
    if not grid:
        return
    if subcommand != "fit":
        sys.exit("Grid search is only supported for the `fit` subcommand; remove the `grid` section.")
    if not is_login_node() and not is_compute_node():
        sys.exit("Grid search launches one SLURM job per combination, so it cannot run outside SLURM; remove the `grid` section to run locally.")


def _submit_one(slurm: SlurmConfig, python_args: list[str], combo: dict[str, Any]) -> None:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    script = env.get_template("slurm.jinja").render(
        **asdict(slurm),
        chdir=str(Path.cwd()),
        activate_cmd="mamba activate ./.env",
        run_cmd=f"srun python {sys.argv[0]} {shlex.join(python_args)}",
    )

    res = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {res.returncode}):\n{res.stderr}\n\nGenerated script:\n{script}"
        )
    job_id = res.stdout.strip().split()[-1]
    overrides = ", ".join(f"{path}={value}" for path, value in combo.items())
    print(f"Submitted batch job {job_id}" + (f" ({overrides})" if overrides else ""))
