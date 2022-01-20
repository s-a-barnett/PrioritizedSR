#!/bin/bash
#SBATCH --job-name=sequential_no_act
#SBATCH --output=outputs/slurm-%A.%a.out
#SBATCH --error=outputs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:59:59
#SBATCH --array=0-1
#SBATCH --mail-type=all
#SBATCH --mail-user=samuelab@princeton.edu

module load anaconda3
cd /home/samuelab/Repos/PrioritizedSR/experiments/
# mkdir -p /tigress/samuelab/PrioritizedSR/logs/sequential_no_act/run2/

ITERS_PER_JOB=36
START=$(( $SLURM_ARRAY_TASK_ID * $ITERS_PER_JOB ))
END=$(( $START + $(($ITERS_PER_JOB - 1)) ))
for i in $(seq $START $END)
do
    srun python train_from_hyperparameters.py sequential_no_act.py hyperparameters/hyperparameters_sequential_no_act.csv $i /scratch/gpfs/samuelab/parsr/results_sequential_no_act.csv
done

