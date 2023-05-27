#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/t1/slurm-%a_t1.log

python t1.py -g 1 -r 2 -seed $SLURM_ARRAY_TASK_ID 
#python readGamma.py -dir t1