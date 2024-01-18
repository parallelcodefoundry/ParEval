#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p serial
#SBATCH -t 04:00:00
#SBATCH -A bhatele-lab-aac
#SBATCH -J omp-codellama-7b-hf_prompted
#SBATCH -o run-outputs/codellama-7b-hf_prompted-omp-%A.out

# settings
HASH="a8724ee8"
MODEL="codellama-7b-hf_prompted"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/omp.json"
SCRATCH_DIR=/tmp

module purge
ml python gcc
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

python run-all.py \
    $GENERATED_PROMPTS \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models omp \
    --early-exit-runs \
    --log info