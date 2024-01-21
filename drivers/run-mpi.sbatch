#!/bin/bash
#SBATCH -n 512
#SBATCH -t 08:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J mpi-gpt-3.5
#SBATCH -o run-outputs/gpt-3.5-mpi-%A.out

# settings
HASH="a8724ee8"
MODEL="gpt-3.5"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/mpi.json"
SCRATCH_DIR=~/scratch/.tmp

module purge
ml python gcc openmpi
export OMPI_MCA_opal_warn_on_missing_libcuda=0

python run-all.py \
    $OUTPUT \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models mpi \
    --early-exit-runs \
    --problem-type geometry \
    --overwrite \
    --run-timeout 60 \
    --log info