#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -t 05:00:00
#SBATCH -A bhatele-lab-aac
#SBATCH -J cuda
#SBATCH -o cuda-%A.out

# settings
GENERATED_PROMPTS=../results/a8724ee8/output_a8724ee8_bigcode--starcoderbase_prompted_temp0.2.json
SCRATCH_DIR=~/scratch/.tmp

module purge
ml python gcc/11.3.0 cuda/12.1.1/gcc/11.3.0/

python run-all.py \
    $GENERATED_PROMPTS \
    --output $GENERATED_PROMPTS \
    --scratch-dir $SCRATCH_DIR \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models cuda \
    --early-exit-runs \
    --log info