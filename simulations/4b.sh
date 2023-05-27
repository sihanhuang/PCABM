#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/4b/slurm-%a_4b.log

python 4b.py -n 200 -seed $SLURM_ARRAY_TASK_ID 
#python readInit.py -dir 4b