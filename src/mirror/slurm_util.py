import os
import re
from dataclasses import dataclass
from typing import Optional

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
    submit: bool = True
    time: str = "01:00:00"
    ntasks_per_node: Optional[int] = None
    gpus_per_node: Optional[int] = None
    mem_per_cpu: str = "128G"
    output: str = "slurm_logs/%j.out"
    open_mode: str = "append"
    signal: str = "SIGHUP@90"
    requeue: bool = True