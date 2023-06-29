#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/dev/null

logpath="../output/3b"
mkdir -p $logpath
logfile="$logpath/slurm-${SLURM_ARRAY_TASK_ID}_3b.log"

python 3b.py -n 200 -g 1.2 -seed $SLURM_ARRAY_TASK_ID > ${logfile}
#python readRes.py -dir 3b
