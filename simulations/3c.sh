#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/3c/slurm-%a_3c.log

python 3c.py -n 200 -r 5 -seed $SLURM_ARRAY_TASK_ID 
#python readRes.py -dir 3c