# PCGBench

This repo contains the Parallel Code Generation Benchmark (PCGBench) for
evaluating LLMs at writing parallel code. See the [PCGBench
Leaderboard](https://pssg.cs.umd.edu/blog/2024/pareval/) for up-to-date results
on different LLMs. The organization of the repo is as follows.

- `prompts/` -- the prompts in PCGBench alongside some utility scripts
- `generate/` -- scripts for generating LLM outputs
- `drivers/` -- scripts to evaluate LLM outputs
- `analysis/` -- scripts to analyze driver results and compute metrics
- `tpl/` -- git submodule dependencies

Each subdirectory has further documentation on its contents. The general
workflow is to use `generate/generate.py` to get LLM outputs, run
`drivers/run-all.py` to evaluate outputs, and `analysis/metrics.py` to
postprocess the results.