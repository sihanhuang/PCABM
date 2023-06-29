#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/dev/null

logpath="../output/3c"
mkdir -p $logpath
logfile="$logpath/slurm-${SLURM_ARRAY_TASK_ID}_3c.log"

python 3c.py -n 200 -r 5 -seed $SLURM_ARRAY_TASK_ID > ${logfile}
#python readRes.py -dir 3c
