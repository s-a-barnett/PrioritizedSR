#!/bin/bash
#SBATCH --job-name=hypersweep
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:59:59
#SBATCH --array=0-499
#SBATCH --mail-type=all
#SBATCH --mail-user=t-sbarnett@microsoft.com

module load anaconda3
srun python train_from_hyperparameters.py $SLURM_ARRAY_TASK_ID

