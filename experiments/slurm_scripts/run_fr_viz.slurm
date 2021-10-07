#!/bin/bash
#SBATCH --job-name=fr_viz
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
mkdir -p /tigress/samuelab/PrioritizedSR/logs/four_rooms/

START=$(( $SLURM_ARRAY_TASK_ID * 3 ))
END=$(( $START + 2 ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py four_rooms.py hyperparameters/hyperparameters_fr_viz.csv $i /tigress/samuelab/PrioritizedSR/logs/four_rooms/results_four_rooms.csv
done

