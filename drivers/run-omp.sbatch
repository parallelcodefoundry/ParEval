#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p serial
#SBATCH -t 05:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J omp-magicoder-s-ds-6.7b
#SBATCH -o run-outputs/magicoder-s-ds-6.7b-omp-%A.out

# settings
HASH="015cff6f"
MODEL="magicoder-s-ds-6.7b"
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