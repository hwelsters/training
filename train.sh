#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 10            # 10 cores per job
#SBATCH -t 0-10:00:00   # 2 hours
#SBATCH -p highmem      # partition
#SBATCH -q grp_pshakari # QOS
#SBATCH --mem 100g       # 100 GB memory per job
#SBATCH -o job_logs/slurm.%A_%a.out
#SBATCH -e job_logs/slurm.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --export=NONE

source /scratch/nngu2/Sam_LoRA/Sam_LoRA/venv/bin/activate

cd /scratch/nngu2/Sam_LoRA/Sam_LoRA
pip install -r requirements.txt

python train.py