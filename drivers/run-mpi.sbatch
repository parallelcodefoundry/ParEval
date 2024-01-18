#!/bin/bash
#SBATCH -N 8
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -A bhatele-lab-aac
#SBATCH -J mpi
#SBATCH -o mpi-%A.out

# settings
GENERATED_PROMPTS=../results/a8724ee8/output_a8724ee8_bigcode--starcoderbase_prompted_temp0.2.json
SCRATCH_DIR=~/scratch/.tmp

module purge
ml python gcc openmpi
export OMPI_MCA_opal_warn_on_missing_libcuda=0

python run-all.py \
    $GENERATED_PROMPTS \
    --output $GENERATED_PROMPTS \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models mpi \
    --early-exit-runs \
    --log info