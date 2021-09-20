#!/bin/bash
#SBATCH --job-name=four_rooms
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:30:00
#SBATCH --array=0-199
#SBATCH --mail-type=all
#SBATCH --mail-user=samuelab@princeton.edu

module load anaconda3
cd /home/samuelab/Repos/PrioritizedSR/experiments/
mkdir -p /tigress/samuelab/PrioritizedSR/logs/four_rooms/

srun python train_from_hyperparameters.py four_rooms.py hyperparameters/hyperparameters_four_rooms.csv $SLURM_ARRAY_TASK_ID /tigress/samuelab/PrioritizedSR/logs/four_rooms/results_four_rooms.csv

# START=$(( $SLURM_ARRAY_TASK_ID * 7 ))
# END=$(( $START + 6 ))
# for i in $(seq $START $END)
# do
#     srun python train_from_hyperparameters.py four_rooms.py $i
# done

