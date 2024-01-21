#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p serial
#SBATCH -t 05:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J serial-gpt-3.5
#SBATCH -o run-outputs/gpt-3.5-serial-%A.out

# settings
HASH="a8724ee8"
MODEL="gpt-3.5"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/serial.json"
SCRATCH_DIR=/tmp

module purge
ml python gcc

python run-all.py \
    $GENERATED_PROMPTS \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models serial \
    --early-exit-runs \
    --log info