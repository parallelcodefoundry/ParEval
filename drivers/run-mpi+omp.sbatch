#!/bin/bash
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J mpi+omp-codellama-34b-hf_prompted
#SBATCH -o run-outputs/codellama-34b-hf_prompted-mpi+omp-%A.out

# settings
HASH="a8724ee8"
MODEL="codellama-34b-hf_prompted"
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
    $OUTPUT \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models mpi+omp \
    --early-exit-runs \
    --problem-type geometry \
    --overwrite \
    --run-timeout 60 \
    --log info