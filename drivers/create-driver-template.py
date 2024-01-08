""" Helper script to create template files for implementing a new driver.
    author: Daniel Nichols
    date: November 2023
"""
# std imports
from argparse import ArgumentParser
import json
import os
from string import Template

# READ IN ARGUMENTS
parser = ArgumentParser(description=__doc__)
parser.add_argument("prompts", help="Path to prompts.json file")
parser.add_argument("driver_name", help="Name of the driver to create")
parser.add_argument("output_dir", help="Path to output directory")
args = parser.parse_args()


CPU_CC_FMT = """// Driver for $prompt_name for Serial, OpenMP, MPI, and MPI+OpenMP
$prompt_as_comment

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {

};

void reset(Context *ctx) {

}

Context *init() {
    Context *ctx = new Context();



    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {

}

void best(Context *ctx) {
    correct
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input

        // compute correct result

        // compute test result
        SYNC();

        if (IS_ROOT(rank) && ) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
"""
CPU_CC_TMPL = Template(CPU_CC_FMT)

KOKKOS_CC_FMT = """// Driver for $prompt_name for Kokkos
$prompt_as_comment

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {

};

void reset(Context *ctx) {

}

Context *init() {
    Context *ctx = new Context();



    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {

}

void best(Context *ctx) {
    correct
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input

        // compute correct result

        // compute test result

        if () {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
"""
KOKKOS_CC_TMPL = Template(KOKKOS_CC_FMT)

GPU_CC_FMT = """// Driver for $prompt_name for CUDA and HIP
$prompt_as_comment

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.cuh"   // code generated by LLM


#if defined(USE_CUDA)
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#endif

struct Context {

    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {

}

Context *init() {
    Context *ctx = new Context();

    ctx->N = 100000;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    <<<gridSize, blockSize>>>
}

void best(Context *ctx) {
    correct
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input

        // compute correct result

        // compute test result

        SYNC();

        // copy back

        if () {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
"""
GPU_CC_TMPL = Template(GPU_CC_FMT)

# READ IN PROMPTS
with open(args.prompts, "r") as f:
    prompts = json.load(f)

# FIND PROMPTS FOR DRIVER
prompts = list(filter(lambda x: x["name"] == args.driver_name, prompts))

# CREATE OUTPUT_DIR IF IT DOESN'T EXIST
os.makedirs(args.output_dir, exist_ok=True)

# CHECK IF ANY OF baseline.hpp, cpu.cc, kokkos.cc, or gpu.cu ALREADY EXIST
for filename in ["baseline.hpp", "cpu.cc", "kokkos.cc", "gpu.cu"]:
    if os.path.exists(os.path.join(args.output_dir, filename)):
        raise RuntimeError(f"File {filename} already exists in {args.output_dir}")

# CREATE baseline.hpp
serial_prompt = list(filter(lambda x: x["parallelism_model"] == "serial", prompts))
assert len(serial_prompt) == 1, "Too many serial prompts"
serial_prompt = serial_prompt[0]["prompt"]
baseline_hpp_fpath = os.path.join(args.output_dir, "baseline.hpp")
with open(baseline_hpp_fpath, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <vector>\n\n")
    f.write(serial_prompt)
    f.write("\n\t\n}")

# CREATE cpu.cc
cpu_cc_fpath = os.path.join(args.output_dir, "cpu.cc")
with open(cpu_cc_fpath, "w") as f:
    serial_prompt_as_comment = '// ' + serial_prompt.replace("\n", "\n// ")
    f.write(CPU_CC_TMPL.substitute(prompt_name=args.driver_name, prompt_as_comment=serial_prompt_as_comment))

# CREATE kokkos.cc
kokkos_cc_fpath = os.path.join(args.output_dir, "kokkos.cc")
kokkos_prompt = list(filter(lambda x: x["parallelism_model"] == "kokkos", prompts))
assert len(kokkos_prompt) == 1, "Too many kokkos prompts"
kokkos_prompt = kokkos_prompt[0]["prompt"]
with open(kokkos_cc_fpath, "w") as f:
    kokkos_prompt_as_comment = '// ' + kokkos_prompt.replace("\n", "\n// ")
    f.write(KOKKOS_CC_TMPL.substitute(prompt_name=args.driver_name, prompt_as_comment=kokkos_prompt_as_comment))

# CREATE gpu.cu
gpu_cc_fpath = os.path.join(args.output_dir, "gpu.cu")
gpu_prompt = list(filter(lambda x: x["parallelism_model"] == "cuda", prompts))
assert len(gpu_prompt) == 1, "Too many gpu prompts"
gpu_prompt = gpu_prompt[0]["prompt"]
with open(gpu_cc_fpath, "w") as f:
    gpu_prompt_as_comment = '// ' + gpu_prompt.replace("\n", "\n// ")
    f.write(GPU_CC_TMPL.substitute(prompt_name=args.driver_name, prompt_as_comment=gpu_prompt_as_comment))
