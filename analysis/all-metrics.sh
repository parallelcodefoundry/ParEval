#!/bin/sh

# main metrics
python metrics.py ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-7B --output ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-13B --output ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-34B --output ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/starcoderbase_prompted_temp0.2/results.csv --model-name StarCoderBase --output ../results/a8724ee8/starcoderbase_prompted_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/phind-v2_prompted_temp0.2/results.csv --model-name Phind-V2 --output ../results/a8724ee8/phind-v2_prompted_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/gpt-3.5_temp0.2/results.csv --model-name GPT-3.5 --output ../results/a8724ee8/gpt-3.5_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/gpt-4_temp0.2/results.csv --model-name GPT-4 --output ../results/a8724ee8/gpt-4_temp0.2/metrics.csv
(cd ../results/a8724ee8 && head -n 1 codellama-7b-hf_prompted_temp0.2/metrics.csv > all-metrics.csv && tail -n+2 -q */metrics.csv >> all-metrics.csv)

# mpi scaling metrics
python metrics-scaling.py ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-7B -k 1 -n 1 2 4 8 16 32 64 128 256 512 --execution-model mpi --output ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/metrics-scaling-mpi.csv
python metrics-scaling.py ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-13B -k 1 -n 1 2 4 8 16 32 64 128 256 512 --execution-model mpi --output ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/metrics-scaling-mpi.csv
python metrics-scaling.py ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-34B -k 1 -n 1 2 4 8 16 32 64 128 256 512 --execution-model mpi --output ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/metrics-scaling-mpi.csv
python metrics-scaling.py ../results/a8724ee8/starcoderbase_prompted_temp0.2/results.csv --model-name StarCoderBase -k 1 -n 1 2 4 8 16 32 64 128 256 512 --execution-model mpi --output ../results/a8724ee8/starcoderbase_prompted_temp0.2/metrics-scaling-mpi.csv
python metrics-scaling.py ../results/a8724ee8/phind-v2_prompted_temp0.2/results.csv --model-name Phind-V2 -k 1 -n 1 2 4 8 16 32 64 128 256 512 --execution-model mpi --output ../results/a8724ee8/phind-v2_prompted_temp0.2/metrics-scaling-mpi.csv
python metrics-scaling.py ../results/a8724ee8/gpt-3.5_temp0.2/results.csv --model-name GPT-3.5 -k 1 -n 1 2 4 8 16 32 64 128 256 512 --execution-model mpi --output ../results/a8724ee8/gpt-3.5_temp0.2/metrics-scaling-mpi.csv
python metrics-scaling.py ../results/a8724ee8/gpt-4_temp0.2/results.csv --model-name GPT-4 -k 1 -n 1 2 4 8 16 32 64 128 256 512 --execution-model mpi --output ../results/a8724ee8/gpt-4_temp0.2/metrics-scaling-mpi.csv
(cd ../results/a8724ee8 && head -n 1 codellama-7b-hf_prompted_temp0.2/metrics-scaling-mpi.csv > all-metrics-scaling-mpi.csv && tail -n+2 -q */metrics-scaling-mpi.csv >> all-metrics-scaling-mpi.csv)

# omp scaling metrics
python metrics-scaling.py ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-7B -k 1 -n 1 2 4 8 16 32 --execution-model omp --output ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/metrics-scaling-omp.csv
python metrics-scaling.py ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-13B -k 1 -n 1 2 4 8 16 32 --execution-model omp --output ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/metrics-scaling-omp.csv
python metrics-scaling.py ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-34B -k 1 -n 1 2 4 8 16 32 --execution-model omp --output ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/metrics-scaling-omp.csv
python metrics-scaling.py ../results/a8724ee8/starcoderbase_prompted_temp0.2/results.csv --model-name StarCoderBase -k 1 -n 1 2 4 8 16 32 --execution-model omp --output ../results/a8724ee8/starcoderbase_prompted_temp0.2/metrics-scaling-omp.csv
python metrics-scaling.py ../results/a8724ee8/phind-v2_prompted_temp0.2/results.csv --model-name Phind-V2 -k 1 -n 1 2 4 8 16 32 --execution-model omp --output ../results/a8724ee8/phind-v2_prompted_temp0.2/metrics-scaling-omp.csv
python metrics-scaling.py ../results/a8724ee8/gpt-3.5_temp0.2/results.csv --model-name GPT-3.5 -k 1 -n 1 2 4 8 16 32 --execution-model omp --output ../results/a8724ee8/gpt-3.5_temp0.2/metrics-scaling-omp.csv
python metrics-scaling.py ../results/a8724ee8/gpt-4_temp0.2/results.csv --model-name GPT-4 -k 1 -n 1 2 4 8 16 32 --execution-model omp --output ../results/a8724ee8/gpt-4_temp0.2/metrics-scaling-omp.csv
(cd ../results/a8724ee8 && head -n 1 codellama-7b-hf_prompted_temp0.2/metrics-scaling-omp.csv > all-metrics-scaling-omp.csv && tail -n+2 -q */metrics-scaling-omp.csv >> all-metrics-scaling-omp.csv)

# kokkos scaling metrics
python metrics-scaling.py ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-7B -k 1 -n 1 2 4 8 16 32 --execution-model kokkos --output ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/metrics-scaling-kokkos.csv
python metrics-scaling.py ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-13B -k 1 -n 1 2 4 8 16 32 --execution-model kokkos --output ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/metrics-scaling-kokkos.csv
python metrics-scaling.py ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-34B -k 1 -n 1 2 4 8 16 32 --execution-model kokkos --output ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/metrics-scaling-kokkos.csv
python metrics-scaling.py ../results/a8724ee8/starcoderbase_prompted_temp0.2/results.csv --model-name StarCoderBase -k 1 -n 1 2 4 8 16 32 --execution-model kokkos --output ../results/a8724ee8/starcoderbase_prompted_temp0.2/metrics-scaling-kokkos.csv
python metrics-scaling.py ../results/a8724ee8/phind-v2_prompted_temp0.2/results.csv --model-name Phind-V2 -k 1 -n 1 2 4 8 16 32 --execution-model kokkos --output ../results/a8724ee8/phind-v2_prompted_temp0.2/metrics-scaling-kokkos.csv
python metrics-scaling.py ../results/a8724ee8/gpt-3.5_temp0.2/results.csv --model-name GPT-3.5 -k 1 -n 1 2 4 8 16 32 --execution-model kokkos --output ../results/a8724ee8/gpt-3.5_temp0.2/metrics-scaling-kokkos.csv
python metrics-scaling.py ../results/a8724ee8/gpt-4_temp0.2/results.csv --model-name GPT-4 -k 1 -n 1 2 4 8 16 32 --execution-model kokkos --output ../results/a8724ee8/gpt-4_temp0.2/metrics-scaling-kokkos.csv
(cd ../results/a8724ee8 && head -n 1 codellama-7b-hf_prompted_temp0.2/metrics-scaling-kokkos.csv > all-metrics-scaling-kokkos.csv && tail -n+2 -q */metrics-scaling-kokkos.csv >> all-metrics-scaling-kokkos.csv)