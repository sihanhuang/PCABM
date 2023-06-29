#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/dev/null

logpath="../output/6"
mkdir -p $logpath
logfile="$logpath/slurm-${SLURM_ARRAY_TASK_ID}_6.log"

python 6.py -n 500 -g 2 -r 4 -seed $SLURM_ARRAY_TASK_ID > ${logfile}
#python readCov.py -dir 6
