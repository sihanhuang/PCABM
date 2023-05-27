#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=../output/ta2/slurm-%a_ta2.log

python ta2.py -n 500 -g 2 -r 4 -seed $SLURM_ARRAY_TASK_ID 
#python readsv.py -dir ta2