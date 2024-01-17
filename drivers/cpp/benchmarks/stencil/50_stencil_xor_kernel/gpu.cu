// Driver for 50_stencil_xor_kernel for CUDA and HIP
// /* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
//    Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
//    input and output are NxN grids of ints in row-major.
//    Use CUDA to compute in parallel. The kernel is launched on an NxN grid of threads.
//    Example:
//
//    input: [[0, 1, 1, 0],
//            [1, 0, 0, 0],
//            [0, 0, 0, 0],
//            [0, 1, 0, 0]
//    output: [[0, 0, 1, 1],
//             [1, 0, 0, 1],
//             [0, 0, 1, 0],
//             [1, 0, 1, 0]]
// */
// __global__ void cellsXOR(const int *input, int *output, size_t N) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.cuh"   // code generated by LLM


#if defined(USE_CUDA)
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#endif

struct Context {
    std::vector<int> h_input, h_output;
    int *d_input, *d_output;

    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    FILL_RAND(ctx->h_input, 0, 1);
    COPY_H2D(ctx->d_input, ctx->h_input.data(), ctx->N * ctx->N);
    std::fill(ctx->h_output.begin(), ctx->h_output.end(), 0);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads
    ctx->h_input.resize(ctx->N * ctx->N);
    ctx->h_output.resize(ctx->N * ctx->N);
    ALLOC(ctx->d_input, ctx->N * ctx->N * sizeof(int));
    ALLOC(ctx->d_output, ctx->N * ctx->N * sizeof(int));

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    cellsXOR<<<ctx->gridSize, ctx->blockSize>>>(ctx->d_input, ctx->d_output, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctCellsXOR(ctx->h_input, ctx->h_output, ctx->N);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    std::vector<int> h_input(TEST_SIZE * TEST_SIZE), correct(TEST_SIZE * TEST_SIZE), test(TEST_SIZE * TEST_SIZE);

    int *d_input, *d_test;
    ALLOC(d_input, TEST_SIZE * TEST_SIZE * sizeof(int));
    ALLOC(d_test, TEST_SIZE * TEST_SIZE * sizeof(int));

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(h_input, 0, 1);
        std::fill(h_output.begin(), h_output.end(), 0);

        COPY_H2D(d_input, h_input.data(), TEST_SIZE * TEST_SIZE * sizeof(int));

        // compute correct result
        correctCellsXOR(h_input, correct, TEST_SIZE);

        // compute test result
        cellsXOR<<<gridSize, blockSize>>>(d_input, d_test, TEST_SIZE);
        SYNC();

        // copy back
        COPY_D2H(test.data(), d_test, TEST_SIZE * TEST_SIZE * sizeof(int));

        if (!std::equal(correct.begin(), correct.end(), test.begin())) {
            FREE(d_input);
            FREE(d_test);
            return false;
        }
    }

    FREE(d_input);
    FREE(d_test);
    return true;
}

void destroy(Context *ctx) {
    FREE(ctx->d_input);
    FREE(ctx->d_output);
    delete ctx;
}
