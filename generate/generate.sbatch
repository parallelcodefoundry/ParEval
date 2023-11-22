#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128000
#SBATCH -t 09:00:00
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -J generate-code
#SBATCH -o generate-%A.out

# settings
MODEL="bigcode/starcoderbase"
TEMP=0.2
TOPP=0.95
MAX_NEW_TKNS=2048
SAMPLES_PER_PROMPT=20
BATCH_SIZE=16
hash=$(md5sum ../prompts/generation-prompts.json | cut -d' ' -f1)
OUTPUT="../outputs/output_${hash:0:8}_${MODEL//\//--}_temp${TEMP}.json"
echo "Writing to $OUTPUT"

# setup
ml cuda/11.8.0
source .env/bin/activate
export HF_HOME=~/scratch/.cache/huggingface
export OMP_NUM_THREADS=8

# generate
python generate.py \
    --model $MODEL \
    --prompts ../prompts/generation-prompts.json \
    --output $OUTPUT \
    --temperature $TEMP \
    --top_p $TOPP \
    --do_sample \
    --max_new_tokens $MAX_NEW_TKNS \
    --num_samples_per_prompt $SAMPLES_PER_PROMPT \
    --batch_size $BATCH_SIZE