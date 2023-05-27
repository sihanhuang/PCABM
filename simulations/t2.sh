#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/t2/slurm-%a_t2.log

python t2.py -n 1000 -g 1 -r 5 -seed $SLURM_ARRAY_TASK_ID 
#python readK.py -dir t2