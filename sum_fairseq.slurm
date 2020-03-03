#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 4

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant
# And finally run the job​

srun python3 sumeval.py 
