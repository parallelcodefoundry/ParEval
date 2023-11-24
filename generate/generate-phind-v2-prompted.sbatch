#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=128000
#SBATCH -t 23:59:59
#SBATCH -A m2404
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -J generate-phind-v2-prompted
#SBATCH -o generate-phind-v2-prompted-%A.out

# settings
MODEL="Phind/Phind-CodeLlama-34B-v2"
TEMP=0.2
TOPP=0.95
MAX_NEW_TKNS=1024
SAMPLES_PER_PROMPT=20
BATCH_SIZE=8
hash=$(md5sum ../prompts/generation-prompts.json | cut -d' ' -f1)
OUTPUT="../outputs/output_${hash:0:8}_${MODEL//\//--}_prompted_temp${TEMP}.json"
echo "Writing to $OUTPUT"
echo "model=$MODEL   MAX_NEW_TKNS=$MAX_NEW_TKNS   SAMPLES_PER_PROMPT=$SAMPLES_PER_PROMPT   BATCH_SIZE=$BATCH_SIZE"

# setup
ml cuda/11.8.0
source .env/bin/activate
export HF_HOME=/pscratch/sd/d/dnicho/.cache/huggingface
export OMP_NUM_THREADS=8
#export SLURM_CPU_BIND="cores"

# generate
srun python generate.py \
    --model $MODEL \
    --prompts ../prompts/generation-prompts.json \
    --output $OUTPUT \
    --temperature $TEMP \
    --top_p $TOPP \
    --do_sample \
    --max_new_tokens $MAX_NEW_TKNS \
    --num_samples_per_prompt $SAMPLES_PER_PROMPT \
    --batch_size $BATCH_SIZE \
    --prompted