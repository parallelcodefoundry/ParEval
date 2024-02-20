#!/bin/sh

HASH="015cff6f"

# combine all
cd $HASH

for dir in $(ls -d */); do
    cd $dir
    echo "Combining $dir"
    python ../../combine.py \
        cuda.json hip.json kokkos.json mpi.json mpi+omp.json omp.json serial.json \
        --check ../../../prompts/generation-prompts.json \
        --output results.json
    python ../../../analysis/create-dataframe.py \
        results.json \
        --output results.csv
    cd ..
done

cd ..