#!/bin/bash
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J hip-codellama-34b-hf_prompted
#SBATCH -o run-outputs/codellama-34b-hf_prompted-hip-{{id}}.out

# settings
HASH="a8724ee8"
MODEL="codellama-34b-hf_prompted"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/hip.json"
SCRATCH_DIR=/tmp

module purge
ml python rocm/5.7.0 flux_wrappers/0.1

python run-all.py \
    $OUTPUT \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models hip \
    --early-exit-runs \
    --problem-type geometry \
    --overwrite \
    --log info