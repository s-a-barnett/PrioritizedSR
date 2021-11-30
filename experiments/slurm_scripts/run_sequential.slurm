#!/bin/bash
#SBATCH --job-name=sequential
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-199
#SBATCH --mail-type=all
#SBATCH --mail-user=samuelab@princeton.edu

module load anaconda3
cd /home/samuelab/Repos/PrioritizedSR/experiments/
mkdir -p /tigress/samuelab/PrioritizedSR/logs/sequential/run6/

START=$(( $SLURM_ARRAY_TASK_ID * 6 ))
END=$(( $START + 5 ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py sequential.py hyperparameters/hyperparameters_sequential.csv $i /tigress/samuelab/PrioritizedSR/logs/sequential/run6/results_sequential.csv
done

