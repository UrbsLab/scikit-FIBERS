#!/bin/bash
#SBATCH -p defq
#SBATCH -n 4
#SBATCH --job-name=EvoSTAR
#SBATCH --mem=4G
#SBATCH -o run_history/logs.o
#SBATCH -e run_history/logs.e
srun python evostar_experiment.py --iterations 1000 --number-of-features 75