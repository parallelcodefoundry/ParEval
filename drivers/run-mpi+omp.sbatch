#!/bin/bash
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -t 05:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J mpi+omp-magicoder-s-ds-6.7b
#SBATCH -o run-outputs/magicoder-s-ds-6.7b-mpi+omp-%A.out

# settings
HASH="015cff6f"
MODEL="magicoder-s-ds-6.7b"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/mpi+omp.json"
SCRATCH_DIR=~/scratch/.tmp

module purge
ml python gcc openmpi
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMPI_MCA_opal_warn_on_missing_libcuda=0

python run-all.py \
    $GENERATED_PROMPTS \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models mpi+omp \
    --early-exit-runs \
    --run-timeout 60 \
    --log info