#!/bin/bash
#SBATCH --job-name=policy_reval
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:59:59
#SBATCH --array=0-199
#SBATCH --mail-type=all
#SBATCH --mail-user=samuelab@princeton.edu

module load anaconda3
cd /home/samuelab/Repos/PrioritizedSR/experiments/
mkdir -p /tigress/samuelab/PrioritizedSR/logs/policy_reval/

START=$(( $SLURM_ARRAY_TASK_ID * 10 ))
END=$(( $START + 9 ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py policy_reval.py hyperparameters/hyperparameters_rm.csv $SLURM_ARRAY_TASK_ID /tigress/samuelab/PrioritizedSR/logs/policy_reval/results_policy_reval.csv
done

