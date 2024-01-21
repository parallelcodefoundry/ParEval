#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p serial
#SBATCH -t 08:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J kokkos-gpt-3.5
#SBATCH -o run-outputs/gpt-3.5-kokkos-%A.out

# settings
HASH="a8724ee8"
MODEL="gpt-3.5"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/kokkos.json"
SCRATCH_DIR=/tmp

module purge
ml python gcc/11.3.0 cmake/gcc/11.3.0
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

python run-all.py \
    $OUTPUT \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models kokkos \
    --early-exit-runs \
    --log info