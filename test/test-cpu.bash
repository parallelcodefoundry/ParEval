#!/bin/bash
# Uses the baseline implementations to test the CPU capabilities of the system.

# usage: bash test/test-cpu.bash <?problem>
if [ $# -eq 0 ]; then
    echo "No problem specified. Using default: 'all'."
    PROBLEM_ARG=""
else
    PROBLEM_ARG="--problem $1"
fi

# First, use the baseline implementations to mimic LLM outputs.
python prompts/create-serial-tests.py drivers/cpp/benchmarks prompts/generation-prompts.json serial-generations.json

# make sure the model drivers are built
cd drivers
cd cpp
make
cd ..

# Run the drivers using these generations
python run-all.py \
    ../serial-generations.json \
    --output results.json \
    --launch-configs launch-configs.json \
    --problem-sizes problem-sizes.json \
    --yes-to-all \
    --include-models serial \
    ${PROBLEM_ARG} \
    --build-timeout 60 \
    --run-timeout 120 \
    --log info