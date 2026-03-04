import os
import re
from dataclasses import dataclass
from typing import Optional, Literal

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