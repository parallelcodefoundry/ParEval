#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p serial
#SBATCH -t 06:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J kokkos-codellama-13b-hf_prompted
#SBATCH -o run-outputs/codellama-13b-hf_prompted-kokkos-%A.out

# settings
HASH="a8724ee8"
MODEL="codellama-13b-hf_prompted"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/kokkos.json"
SCRATCH_DIR=/tmp

module purge
ml python gcc/11.3.0 cmake/gcc/11.3.0
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

python run-all.py \
    $GENERATED_PROMPTS \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models kokkos \
    --early-exit-runs \
    --log info