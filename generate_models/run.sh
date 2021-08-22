#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J saldata
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=[v100]
#SBATCH --mem=128G
#SBATCH --array=0-7


# ----- Extra variable ----- #
#SBATCH --mail-user=abdullah.alali.1@kaust.edu.sa
#SBATCH --mail-type=ALL


# load module 
module load deepwave/0.0.7
module load madagascar-gpu/3.0.1/gnu6.4.0_cuda10.1


#module load cuda
#module load  anaconda3/4.4.0
#export ENV_PREFIX=$PWD/env
#conda env create -p $PWD/env --file deepwave.yml --force
#source  activate $ENV_PREFIX






#run the application: arguments are istart num_model

echo "This is job $SLURM_JOB_ID with task ID  $SLURM_ARRAY_TASK_ID"

srun python gen_data.py $SLURM_ARRAY_TASK_ID  1000
