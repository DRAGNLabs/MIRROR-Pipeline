import os
import re

def get_job_id():
    array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    if array_job_id is not None:
        array_task_id = os.environ['SLURM_ARRAY_TASK_ID']
        job_id = f'{array_job_id}_{array_task_id}'
    else:
        job_id = os.environ['SLURM_JOB_ID']

    assert re.match('[0-9_-]+', job_id)

    return job_id

