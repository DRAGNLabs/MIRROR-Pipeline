import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Literal

import yaml

from mirror.util import resolve_config_args


def get_job_id():
    array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    if array_job_id is not None:
        array_task_id = os.environ['SLURM_ARRAY_TASK_ID']
        job_id = f'{array_job_id}_{array_task_id}'
    else:
        job_id = os.environ['SLURM_JOB_ID']

    assert re.match('[0-9_-]+', job_id)

    return job_id

@dataclass
class SlurmConfig:
    job_type: Literal["compute", "local", "local-download"] = "compute"
    time: str = "01:00:00"
    nodes: Optional[int] = None
    ntasks_per_node: Optional[int] = None
    gpus_per_node: Optional[str] = None
    mem_per_cpu: str = "128G"
    output: str = "slurm_logs/%j.out"
    open_mode: str = "append"
    signal: str = "SIGHUP@90"
    requeue: bool = True
    qos: Optional[str] = None


def parse_slurm_config(python_args: list[str]) -> SlurmConfig:
    """Build a SlurmConfig from --config files and --slurm.<key>[=value] CLI args.

    Falls back to the `trainer` section's `num_nodes` / `devices` for unset
    `nodes`, `ntasks_per_node`, and `gpus_per_node`.
    """
    sections = _collect_sections(resolve_config_args(python_args), ('slurm', 'trainer'))
    slurm = SlurmConfig(**sections['slurm'])
    trainer = sections['trainer']
    slurm.nodes = slurm.nodes or trainer.get('num_nodes') or 1
    slurm.ntasks_per_node = slurm.ntasks_per_node or trainer.get('devices') or 1
    slurm.gpus_per_node = slurm.gpus_per_node or str(trainer.get('devices') or 1)
    return slurm


def _collect_sections(args: list[str], names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Collect the named top-level keys from --config files and --<name>.<key>[=value] CLI args."""
    out: dict[str, dict[str, Any]] = {n: {} for n in names}
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