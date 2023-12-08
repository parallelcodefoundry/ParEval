#!/bin/bash
#SBATCH -A m2404
#SBATCH -C cpu
#SBATCH -t 23:59:59
#SBATCH -q regular
#SBATCH -J bin-the-stack
#SBATCH -o bin-the-stack-%A.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1


export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

export HF_HOME=/pscratch/sd/d/dnicho/.cache/huggingface

source ../generate/.env/bin/activate

srun --cpu-bind=cores python bin-the-stack.py --batch_size 1000