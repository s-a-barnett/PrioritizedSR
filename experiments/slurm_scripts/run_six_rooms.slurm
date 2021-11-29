#!/bin/bash
#SBATCH --job-name=six_rooms
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
mkdir -p /tigress/samuelab/PrioritizedSR/logs/six_rooms/run3/

START=$(( $SLURM_ARRAY_TASK_ID * 8 ))
END=$(( $START + 7 ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py six_rooms.py hyperparameters/hyperparameters_six_rooms.csv $i /tigress/samuelab/PrioritizedSR/logs/six_rooms/run3/results_six_rooms.csv
done

