#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH -A bhatele-lab-aac
#SBATCH -J serial
#SBATCH -o serial-%A.out

HASH="a8724ee8"
MODEL="starcoderbase_prompted"
TEMP="0.2"

# settings
GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/serial.json"
SCRATCH_DIR=~/scratch/.tmp

module purge
ml python gcc

python run-all.py \
    $GENERATED_PROMPTS \
    --output $GENERATED_PROMPTS \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models serial \
    --early-exit-runs \
    --log info