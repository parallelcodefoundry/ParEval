#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -t 05:00:00
#SBATCH -A bhatele-lab-cmsc
#SBATCH -J cuda-codellama-13b-hf_prompted
#SBATCH -o run-outputs/codellama-13b-hf_prompted-cuda-%A.out

# settings
HASH="a8724ee8"
MODEL="codellama-13b-hf_prompted"
TEMP="0.2"

GENERATED_PROMPTS="../results/${HASH}/${MODEL}_temp${TEMP}/all.json"
OUTPUT="../results/${HASH}/${MODEL}_temp${TEMP}/cuda.json"
SCRATCH_DIR=/tmp

module purge
ml python gcc/11.3.0 cuda/12.1.1/gcc/11.3.0/

python run-all.py \
    $GENERATED_PROMPTS \
    --output $OUTPUT \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models cuda \
    --early-exit-runs \
    --log info