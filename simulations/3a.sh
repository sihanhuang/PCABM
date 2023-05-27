#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/3a/slurm-%a_3a.log

python 3a.py -g 1.2 -r 5 -seed $SLURM_ARRAY_TASK_ID 
#python readRes.py -dir 3a