#!/bin/bash
#SBATCH --job-name=rm_detour
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

ITERS_PER_JOB=1
START=$(( $SLURM_ARRAY_TASK_ID * $ITERS_PER_JOB ))
END=$(( $START + $(($ITERS_PER_JOB - 1)) ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py rm_detour.py hyperparameters/hyperparameters_rm_detour.csv $i /scratch/gpfs/samuelab/parsr/results_rm_detour.csv
done

