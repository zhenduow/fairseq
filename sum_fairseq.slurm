#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 1

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant
# And finally run the job​

srun python3 sumeval.py 
