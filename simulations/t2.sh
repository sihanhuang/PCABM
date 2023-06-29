#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/dev/null

logpath="../output/t2"
mkdir -p $logpath
logfile="$logpath/slurm-${SLURM_ARRAY_TASK_ID}_t2.log"

python t2.py -n 1000 -g 1 -r 5 -seed $SLURM_ARRAY_TASK_ID > ${logfile}
#python readK.py -dir t2
