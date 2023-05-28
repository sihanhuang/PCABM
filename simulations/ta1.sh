#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/ta1/slurm-%a_ta1.log

python ta1.py -n 500 -g 2 -r 4 -seed $SLURM_ARRAY_TASK_ID 
#python readsv.py -dir ta1