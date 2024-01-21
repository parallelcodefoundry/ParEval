#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=128000
#SBATCH -t 23:59:59
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -J generate-starcoderbase
#SBATCH -o generate-starcoderbase-%A.out

# settings
MODEL="bigcode/starcoderbase"
TEMP=0.2
TOPP=0.95
MAX_NEW_TKNS=1024
SAMPLES_PER_PROMPT=20
BATCH_SIZE=16
hash=$(md5sum ../prompts/generation-prompts.json | cut -d' ' -f1)
OUTPUT="../outputs/output_${hash:0:8}_${MODEL//\//--}_temp${TEMP}.json"
CACHE="../outputs/cache/cache_${hash:0:8}_${MODEL//\//--}_temp${TEMP}.jsonl"
echo "Writing to $OUTPUT"
echo "model=$MODEL   MAX_NEW_TKNS=$MAX_NEW_TKNS   SAMPLES_PER_PROMPT=$SAMPLES_PER_PROMPT   BATCH_SIZE=$BATCH_SIZE"

# setup
ml cuda/11.8.0
source .env/bin/activate
export HF_HOME=~/scratch/.cache/huggingface
export OMP_NUM_THREADS=16

# generate
srun python generate.py \
    --model $MODEL \
    --prompts ../prompts/generation-prompts.json \
    --cache $CACHE \
    --restore_from ../outputs/zj_outputs/output_496edbdf_bigcode--starcoderbase_temp0.2.json \
    --output $OUTPUT \
    --temperature $TEMP \
    --top_p $TOPP \
    --do_sample \
    --max_new_tokens $MAX_NEW_TKNS \
    --num_samples_per_prompt $SAMPLES_PER_PROMPT \
    --batch_size $BATCH_SIZE