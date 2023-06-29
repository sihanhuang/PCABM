#!/bin/sh

#SBATCH --array=1-100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/dev/null

logpath="../output/3d"
mkdir -p $logpath
logfile="$logpath/slurm-${SLURM_ARRAY_TASK_ID}_3d.log"

python 3d.py -n 200 -g 1.5 -r 2 -seed $SLURM_ARRAY_TASK_ID > ${logfile}
#python readInit.py -dir 3d
