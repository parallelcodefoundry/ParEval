#!/bin/sh
# A script to help with debugging the drivers

if [ $# -ne 3 ]; then
    echo "Usage: $0 <problem> <model> <log-level>"
    exit 1
fi

OUTPUTS="../outputs/output_800a7a5d_bigcode--starcoderbase_prompted_temp0.2.json"
PROBLEM="$1"
MODEL="$2"
LOG_LEVEL="$3"
shift 3

if [ $LOG_LEVEL == "debug" ]; then
    LOG_ARGS="--log debug --log-build-errors"
else
    LOG_ARGS="--log $LOG_LEVEL"
fi

module purge
if [ $MODEL == "serial" ] || [ $MODEL == "omp" ]; then 
    ml python gcc
elif [ $MODEL == "mpi" ] || [ $MODEL == "mpi+omp" ]; then
    ml python gcc openmpi
elif [ $MODEL == "kokkos" ]; then
    ml python gcc/11.3.0
elif [ $MODEL == "cuda" ]; then
    ml python gcc/11.3.0 cuda/12.1.1/gcc/11.3.0/
else
    echo "Invalid model: $MODEL"
    exit 1
fi

python run-all.py \
    $OUTPUTS \
    -o ./junk.json \
    --scratch-dir ~/scratch/.tmp \
    --launch-configs test-launch-configs.json \
    --yes-to-all \
    --early-exit-runs \
    --run-timeout 15 \
    --problem $PROBLEM \
    --include-models $MODEL \
    $LOG_ARGS \
    $@