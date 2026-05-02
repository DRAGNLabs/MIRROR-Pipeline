import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mirror.slurm_util import SlurmConfig
from mirror.util import resolve_config_args


def submit_slurm_job(python_args: list[str]) -> str | None:
    """Submit a slurm job if `slurm.job_type` is 'compute' on a login node.

    `python_args` is everything after the python script (e.g. `sys.argv[1:]`),
    starting with the subcommand. Returns the slurm job id if a job was
    submitted, otherwise None — the caller should bail out only on a non-None
    return.
    """
    if 'login' not in socket.gethostname():
        return None

    sections = _collect_sections(resolve_config_args(python_args), ('slurm', 'trainer'))
    valid = {f.name for f in fields(SlurmConfig)}
    slurm = SlurmConfig(**{k: v for k, v in sections['slurm'].items() if k in valid})

    if slurm.job_type != "compute":
        return None

    trainer = sections['trainer']
    ctx = asdict(slurm)
    ctx["nodes"] = ctx["nodes"] or trainer.get('num_nodes') or 1
    ctx["ntasks_per_node"] = ctx["ntasks_per_node"] or trainer.get('devices') or 1
    ctx["gpus_per_node"] = ctx["gpus_per_node"] or trainer.get('devices') or 1

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    script = env.get_template("slurm.jinja").render(
        **ctx,
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
    return job_id


def _collect_sections(args: list[str], names: tuple[str, ...]) -> dict[str, dict]:
    """Collect the named top-level keys from --config files and --<name>.<key>[=value] CLI args."""
    out: dict[str, dict] = {n: {} for n in names}
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--config' and i + 1 < len(args):
            with open(args[i + 1]) as f:
                data = yaml.safe_load(f) or {}
            for n in names:
                out[n].update(data.get(n) or {})
            i += 2
            continue
        section = next((n for n in names if a.startswith(f'--{n}.')), None)
        if section is None:
            i += 1
            continue
        key, eq, value = a[len(section) + 3:].partition('=')
        if eq:
            i += 1
        elif i + 1 < len(args) and not args[i + 1].startswith('--'):
            value = args[i + 1]
            i += 2
        else:
            i += 1
            continue
        out[section][key] = yaml.safe_load(value)
    return out
