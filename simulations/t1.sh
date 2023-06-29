#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/dev/null

logpath="../output/t1"
mkdir -p $logpath
logfile="$logpath/slurm-${SLURM_ARRAY_TASK_ID}_t1.log"

python t1.py -g 1 -r 2 -seed $SLURM_ARRAY_TASK_ID > ${logfile}
#python readGamma.py -dir t1
