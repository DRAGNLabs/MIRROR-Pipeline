import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.slurm_util import SlurmConfig, expand_grid, grid_override_args, parse_grid, parse_slurm_config
from mirror.util import is_login_node


def submit_slurm_jobs(python_args: list[str]) -> None:
    """Submit one compute job per `grid` combination (one job if no grid), then exit.

    `python_args` is everything after the python script (e.g. `sys.argv[1:]`),
    starting with the subcommand. No-op when not launching a compute job from a
    login node, so the caller proceeds to run in-process.
    """
    if not is_login_node():
        return

    submitted = False
    for combo in expand_grid(parse_grid(python_args)):
        args = python_args + grid_override_args(combo)
        slurm = parse_slurm_config(args)
        if slurm.job_type == "compute":
            _submit_one(slurm, args)
            submitted = True

    if submitted:
        sys.exit(0)


def _submit_one(slurm: SlurmConfig, python_args: list[str]) -> None:
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
    print(f"Submitted batch job {job_id}")
