#!/bin/bash
###SBATCH -A m2404
###SBATCH -C cpu
#SBATCH -t 72:00:00
###SBATCH -q regular
#SBATCH -p serial
#SBATCH -J bin-the-stack
#SBATCH -o bin-the-stack-%A.out
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --tmp=64000


#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
export OMP_NUM_THREADS=64

export HF_HOME=~/scratch/.cache/huggingface

source ../generate/.env/bin/activate

srun --cpu-bind=cores python bin-the-stack.py --batch_size 256 --chunk_size 1 --num_processes 64 