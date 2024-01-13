#!/bin/sh
# A script to help with debugging the drivers

if [ $# -ne 3 ]; then
    echo "Usage: $0 <problem> <model> <log-level>"
    exit 1
fi

if [[ $(hostname) == nid* ]]; then
    SYSTEM="perlmutter"
else
    SYSTEM="zaratan"
fi

OUTPUTS="../outputs/output_800a7a5d_bigcode--starcoderbase_prompted_temp0.2.json"
PROBLEM="$1"
MODEL="$2"
LOG_LEVEL="$3"
shift 3

# check if PROBLEM is numeric only
if [[ $PROBLEM =~ ^[0-9]+$ ]]; then
    ml python
    PROBLEM=$(python -c "import json; db=json.load(open(\"${OUTPUTS}\", 'r')); print([p['name'] for p in db if p['name'].startswith(\"${PROBLEM}_\")][0])")
    if [ $? -ne 0 ]; then
        echo "Problem $PROBLEM not found in $OUTPUTS"
        exit 1
    fi
fi

if [ $LOG_LEVEL == "debug" ]; then
    LOG_ARGS="--log debug --log-build-errors"
else
    LOG_ARGS="--log $LOG_LEVEL"
fi

if [ $SYSTEM == "zaratan" ]; then
    module purge
fi

if [ $MODEL == "serial" ] || [ $MODEL == "omp" ]; then 
    ml python gcc
elif [ $MODEL == "mpi" ] || [ $MODEL == "mpi+omp" ]; then
    ml python gcc openmpi
elif [ $MODEL == "kokkos" ]; then
    if [ $SYSTEM == "perlmutter" ]; then
        ml python gcc/11.2.0 cmake/3.24.3
    else
        ml python gcc/11.3.0 cmake/gcc/11.3.0
    fi
elif [ $MODEL == "cuda" ]; then
    ml python gcc/11.3.0 cuda/12.1.1/gcc/11.3.0/
else
    echo "Invalid model: $MODEL"
    exit 1
fi

if [ $SYSTEM == "perlmutter" ]; then
    # perlmutter
    SCRATCH_ARGS=""
else
    SCRATCH_ARGS="--scratch-dir ~/scratch/.tmp"
fi

python run-all.py \
    $OUTPUTS \
    -o ./junk.json \
    $SCRATCH_ARGS \
    --launch-configs test-launch-configs.json \
    --yes-to-all \
    --early-exit-runs \
    --run-timeout 15 \
    --problem $PROBLEM \
    --include-models $MODEL \
    $LOG_ARGS \
    $@