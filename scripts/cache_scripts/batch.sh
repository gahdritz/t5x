#!/bin/bash
#
#SBATCH --array 0-19
#SBATCH -p seas_compute
#SBATCH -c 32
#SBATCH -t 1-00:00
#SBATCH --mem 150G
#SBATCH -o scripts/cache_scripts/slurm_outputs/slurm-%j.out
#SBATCH -e scripts/cache_scripts/slurm_outputs/slurm-%j.err

srun scripts/cache_scripts/create_caches.sh 50
