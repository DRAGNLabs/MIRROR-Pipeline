#!/bin/bash --login

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks-per-node=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH --output=slurm_logs/%j.out
#SBATCH --open-mode=append
#SBATCH --signal=SIGHUP@90
#SBATCH --requeue

mamba activate ./.env

srun python src/main.py fit --data.class_path WikitextDataset --data.head 10
