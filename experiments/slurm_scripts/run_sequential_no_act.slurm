#!/bin/bash
#SBATCH --job-name=sequential_no_act
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:01
#SBATCH --array=0-49
#SBATCH --mail-type=all
#SBATCH --mail-user=samuelab@princeton.edu

module load anaconda3
cd /home/samuelab/Repos/PrioritizedSR/experiments/
mkdir -p /tigress/samuelab/PrioritizedSR/logs/sequential_no_act/run2/

START=$(( $SLURM_ARRAY_TASK_ID * 22 ))
END=$(( $START + 21 ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py sequential_no_act.py hyperparameters/hyperparameters_sequential_no_act.csv $i /tigress/samuelab/PrioritizedSR/logs/sequential_no_act/run2/results_sequential_no_act.csv
done

