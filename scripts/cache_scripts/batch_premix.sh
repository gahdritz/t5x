#!/bin/bash
#
#SBATCH --array 0-9
#SBATCH -p seas_compute
#SBATCH -c 32
#SBATCH -t 1-00:00
#SBATCH --mem 150G
#SBATCH -o scripts/cache_scripts/slurm_outputs/slurm-%j-%a.out
#SBATCH -e scripts/cache_scripts/slurm_outputs/slurm-%j-%a.err

srun scripts/cache_scripts/premix.sh
