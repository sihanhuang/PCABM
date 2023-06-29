#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/dev/null

logpath="../output/4a"
mkdir -p $logpath
logfile="$logpath/slurm-${SLURM_ARRAY_TASK_ID}_4a.log"

python 4a.py -r 3 -seed $SLURM_ARRAY_TASK_ID > ${logfile}
#python readRes_score.py -dir 4a
