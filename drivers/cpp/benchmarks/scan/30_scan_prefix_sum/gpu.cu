// Driver for 30_scan_prefix_sum for CUDA and HIP
// /* Compute the prefix sum of the vector x into output.
//    Use CUDA to compute in parallel. The kernel is launched with at least as many threads as elements in x.
//    Example:
//
//    input: [1, 7, 4, 6, 6, 2]
//    output: [1, 8, 12, 18, 24, 26]
// */
// __global__ void prefixSum(const double *x, double *output, size_t N) {

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
    double *x;
    double *output;
    size_t N;
    std::vector<double> h_x;
    std::vector<double> h_output;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, -100.0, 100.0);
    COPY_H2D(ctx->x, ctx->h_x.data(), ctx->N * sizeof(double));
}

Context *init() {
    Context *ctx = new Context();

    ctx->N == 100000;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    ALLOC(ctx->x, ctx->N * sizeof(double));
    ALLOC(ctx->output, ctx->N * sizeof(double));

    ctx->h_x.resize(ctx->N);
    ctx->h_output.resize(ctx->N);

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    prefixSum<<<ctx->gridSize, ctx->blockSize>>>(ctx->x, ctx->output, ctx->N);
}

void best(Context *ctx) {
    correctPrefixSum(ctx->h_x, ctx->h_output);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(TEST_SIZE);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x);

    std::vector<double> correct(TEST_SIZE), input(TEST_SIZE), test(TEST_SIZE);
    double *testDevice, *inputDevice;

    ALLOC(testDevice, correct.size() * sizeof(double));
    ALLOC(inputDevice, correct.size() * sizeof(double));

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(input, -100.0, 100.0);
        COPY_H2D(inputDevice, input.data(), TEST_SIZE * sizeof(double));

        // compute correct result
        correctPrefixSum(input, correct);

        // compute test result
        prefixSum<<<gridSize, blockSize>>>(inputDevice, testDevice, TEST_SIZE);
        SYNC();

        // copy back
        COPY_D2H(test.data(), testDevice, TEST_SIZE * sizeof(double));

        if (!std::equal(correct.begin(), correct.end(), test.begin())) {
            FREE(inputDevice);
            FREE(testDevice);
            return false;
        }
    }

    FREE(inputDevice);
    FREE(testDevice);

    return true;
}

void destroy(Context *ctx) {
    FREE(ctx->x);
    FREE(ctx->output);
    delete ctx;
}
