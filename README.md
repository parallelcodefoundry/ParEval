# PCGBench

This repo contains the Parallel Code Generation Benchmark (PCGBench) for
evaluating the ability of Large Language Models to write parallel code. See the
[PCGBench Leaderboard](https://pssg.cs.umd.edu/blog/2024/pareval/) for
up-to-date results on different LLMs. 

## Overview

The organization of the repo is as follows.

- `prompts/` -- the prompts in PCGBench alongside some utility scripts
- `generate/` -- scripts for generating LLM outputs
- `drivers/` -- scripts to evaluate LLM outputs
- `analysis/` -- scripts to analyze driver results and compute metrics
- `tpl/` -- git submodule dependencies

Each subdirectory has further documentation on its contents. The general
workflow is to use `generate/generate.py` to get LLM outputs, run
`drivers/run-all.py` to evaluate outputs, and `analysis/metrics.py` to
postprocess the results.

## Setup and Installation

A couple core systems software are assumed to be installed: Python >=3.7, a C++
compiler that supports C++17 and OpenMP, Make, CMake, and an MPI implementation.
If you are testing the CUDA and HIP prompts, then you will need access to NVIDIA
and AMD GPUs alongside their respective software stacks.

First, clone the repo.

```sh
git clone --recurse-submodules https://github.com/pssg-int/llms-for-hpc.git
```

Next, you need to build Kokkos (if you want to include it in testing).

```sh
cd tpl/kokkos

mkdir build
cd build

# depending on your system you may need to pass your c++ compiler to CMAKE_CXX_COMPILER
cmake .. -DCMAKE_INSTALL_PREFIX=. -DKokkos_ENABLE_THREADS=ON
make install -j4
```

Finally, you need to install the Python dependencies. `requirements.txt` has
the set of dependencies pinned at the version they were tested with. Other
versions may also work. Note that some of these are only required for parts of
the pipeline i.e. PyTorch and Transformers are only needed for generating LLM
outputs.

```sh
pip install -r requirements.txt
```

## Citing PCGBench

```
@misc{nichols2024large,
      title={Can Large Language Models Write Parallel Code?}, 
      author={Daniel Nichols and Joshua H. Davis and Zhaojun Xie and 
              Arjun Rajaram and Abhinav Bhatele},
      year={2024},
      eprint={2401.12554},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```