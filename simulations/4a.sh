#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/4a/slurm-%a_4a.log

python 4a.py -r 3 -seed $SLURM_ARRAY_TASK_ID 
#python readInit.py -dir 4a