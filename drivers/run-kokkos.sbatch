#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -A bhatele-lab-aac
#SBATCH -J kokkos
#SBATCH -o kokkos-%A.out

# settings
GENERATED_PROMPTS=../results/a8724ee8/output_a8724ee8_bigcode--starcoderbase_prompted_temp0.2.json
SCRATCH_DIR=~/scratch/.tmp

module purge
ml python gcc/11.3.0 cmake/gcc/11.3.0
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

python run-all.py \
    $GENERATED_PROMPTS \
    --output $GENERATED_PROMPTS \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models kokkos \
    --early-exit-runs \
    --log info