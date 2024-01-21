#!/bin/sh


python metrics.py ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-7B --output ../results/a8724ee8/codellama-7b-hf_prompted_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-13B --output ../results/a8724ee8/codellama-13b-hf_prompted_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/results.csv --model-name CodeLlama-34B --output ../results/a8724ee8/codellama-34b-hf_prompted_temp0.2/metrics.csv

python metrics.py ../results/a8724ee8/starcoderbase_prompted_temp0.2/results.csv --model-name StarCoderBase --output ../results/a8724ee8/starcoderbase_prompted_temp0.2/metrics.csv

python metrics.py ../results/a8724ee8/phind-v2_prompted_temp0.2/results.csv --model-name Phind-V2 --output ../results/a8724ee8/phind-v2_prompted_temp0.2/metrics.csv

python metrics.py ../results/a8724ee8/gpt-3.5_temp0.2/results.csv --model-name GPT-3.5 --output ../results/a8724ee8/gpt-3.5_temp0.2/metrics.csv
python metrics.py ../results/a8724ee8/gpt-4_temp0.2/results.csv --model-name GPT-4 --output ../results/a8724ee8/gpt-4_temp0.2/metrics.csv