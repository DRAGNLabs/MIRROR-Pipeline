#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=128G   # memory per CPU core
#SBATCH --output=slurm_logs/%j.out

source env/bin/activate

python src/main.py fit
