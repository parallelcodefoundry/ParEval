#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=128000
#SBATCH -t 01:00:00
#SBATCH -A m2404
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -J throughput-starcoderbase-bs2
#SBATCH -o throughput-starcoderbase-bs2-%A.out

# settings
MODEL="bigcode/starcoderbase"
TEMP=0.2
TOPP=0.95
MAX_NEW_TKNS=1024
SAMPLES_PER_PROMPT=20
BATCH_SIZE=2
WARMUP=5
NUM_BATCHES=25
echo "model=$MODEL   MAX_NEW_TKNS=$MAX_NEW_TKNS   SAMPLES_PER_PROMPT=$SAMPLES_PER_PROMPT   BATCH_SIZE=$BATCH_SIZE"
echo "WARMUP=5   NUM_BATCHES=$NUM_BATCHES"

# setup
source .env/bin/activate
export HF_HOME=/pscratch/sd/d/dnicho/.cache/huggingface
export OMP_NUM_THREADS=8

# generate
srun python throughput.py \
    --model $MODEL \
    --prompts ../prompts/generation-prompts.json \
    --temperature $TEMP \
    --top_p $TOPP \
    --do_sample \
    --max_new_tokens $MAX_NEW_TKNS \
    --num_samples_per_prompt $SAMPLES_PER_PROMPT \
    --batch_size $BATCH_SIZE \
    --warmup $WARMUP \
    --num_batches $NUM_BATCHES