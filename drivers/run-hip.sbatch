#!/bin/bash
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J hip-magicoder-s-ds-6.7b
#SBATCH -o run-outputs/magicoder-s-ds-6.7b-hip-{{id}}.out

# settings
HASH="015cff6f"
MODEL="magicoder-s-ds-6.7b"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/hip.json"
SCRATCH_DIR=/tmp

module purge
ml python rocm/5.7.0 flux_wrappers/0.1

python run-all.py \
    $GENERATED_PROMPTS \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models hip \
    --early-exit-runs \
    --log info