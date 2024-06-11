# ParEval

[![HPDC 2024](https://img.shields.io/badge/Paper-HPDC'24-e87053.svg?style=flat)](https://pssg.cs.umd.edu/assets/papers/2024-06-pareval-hpdc.pdf)&nbsp;[![arXiv](https://img.shields.io/badge/arXiv-2401.12554-b31b1b.svg)](https://arxiv.org/abs/2401.12554)&nbsp;[![GitHub license](https://badgen.net/github/license/parallelcodefoundry/ParEval)](https://github.com/parallelcodefoundry/ParEval/blob/develop/LICENSE)


This repo contains the Parallel Code Evaluation (ParEval) Benchmark for
evaluating the ability of Large Language Models to write parallel code. See the
[ParEval Leaderboard](https://pssg.cs.umd.edu/blog/2024/pareval/) for
up-to-date results on different LLMs.


## Overview

The organization of the repo is as follows.

- `prompts/` -- the prompts in ParEval alongside some utility scripts
- `generate/` -- scripts for generating LLM outputs
- `drivers/` -- scripts to evaluate LLM outputs
- `analysis/` -- scripts to analyze driver results and compute metrics
- `tpl/` -- git submodule dependencies

Each subdirectory has further documentation on its contents. The general
workflow is to use `generate/generate.py` to generate LLM outputs, run
`drivers/run-all.py` to evaluate outputs, and `analysis/metrics.py` to
post-process the results.

## Setup and Installation

A couple core systems software are assumed to be installed: Python >=3.7, a C++
compiler that supports C++17 and OpenMP, Make, CMake, and an MPI implementation.
If you are testing the CUDA and HIP prompts, then you will need access to NVIDIA
and AMD GPUs alongside their respective software stacks.

First, clone the repo.

```sh
git clone --recurse-submodules https://github.com/parallelcodefoundry/ParEval.git
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

## Citing ParEval

```
@misc{nichols2024large,
      title={Can Large Language Models Write Parallel Code?}, 
      author={Daniel Nichols and Joshua H. Davis and Zhaojun Xie and 
              Arjun Rajaram and Abhinav Bhatele},
      year={2024},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      booktitle = {Proceedings of the 33rd International Symposium on High-Performance Parallel and Distributed Computing},
      series = {HPDC '24}
}
```

## License

ParEval is distributed under the terms of the [MIT license](/LICENSE).
