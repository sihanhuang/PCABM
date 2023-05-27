#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/3d/slurm-%a_3d.log

python 3d.py -n 200 -g 1.5 -r 2 -seed $SLURM_ARRAY_TASK_ID 
#python readInit.py -dir 3d