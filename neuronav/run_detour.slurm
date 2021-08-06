#!/bin/bash
#SBATCH --job-name=hypersweep
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-350
#SBATCH --mail-type=all
#SBATCH --mail-user=t-sbarnett@microsoft.com

module load anaconda3
START=$(( $SLURM_ARRAY_TASK_ID * 7 ))
END=$(( $START + 6 ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py four_rooms_detour.py $i
done

