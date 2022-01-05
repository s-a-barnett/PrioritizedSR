#!/bin/bash
#SBATCH --job-name=sequential
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=13:00:00
#SBATCH --array=0-199
#SBATCH --mail-type=all
#SBATCH --mail-user=samuelab@princeton.edu

module load anaconda3
cd /home/samuelab/Repos/PrioritizedSR/experiments/
mkdir -p /tigress/samuelab/PrioritizedSR/logs/sequential/run9/

ITERS_PER_JOB=27
START=$(( $SLURM_ARRAY_TASK_ID * $ITERS_PER_JOB ))
END=$(( $START + $(($ITERS_PER_JOB - 1)) ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py sequential.py hyperparameters/hyperparameters_sequential.csv $i /tigress/samuelab/PrioritizedSR/logs/sequential/run9/results_sequential.csv
done

