// Driver for 21_histogram_bin_0-100 for CUDA and HIP
// /* Vector x contains values between 0 and 100, inclusive. Count the number of
//    values in [0,10), [10, 20), [20, 30), ... and store the counts in `bins`.
//    Use CUDA to compute in parallel. The kernel is initialized with at least as many threads as values in x.
//    Example:
// 
//    input: [7, 32, 95, 12, 39, 32, 11, 71, 70, 66]
//    output: [1, 2, 0, 3, 0, 0, 1, 2, 0, 1]
// */
// __global__ void binsBy10Count(const double *x, size_t N, size_t bins[10]) {

#include <array>
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
    double *d_x;
    size_t *d_bins;
    std::vector<double> h_x;
    std::array<size_t, 10> h_bins;
    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, 0.0, 100.0);
    ctx->h_bins.fill(0);

    COPY_H2D(ctx->d_x, ctx->h_x.data(), ctx->N * sizeof(double));
    COPY_H2D(ctx->d_bins, ctx->h_bins.data(), 10 * sizeof(size_t));
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = 1 << 18;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    ctx->h_x.resize(ctx->N);
    ALLOC(ctx->d_x, ctx->N * sizeof(double));
    ALLOC(ctx->d_bins, 10 * sizeof(size_t));

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    binsBy10Count<<<ctx->gridSize, ctx->blockSize>>>(ctx->d_x, ctx->N, ctx->d_bins);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctBinsBy10Count(ctx->h_x, ctx->h_bins);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    std::vector<double> h_x(TEST_SIZE);
    std::array<size_t, 10> correct, test;

    double *d_x;
    size_t *d_bins;
    ALLOC(d_x, TEST_SIZE * sizeof(double));
    ALLOC(d_bins, 10 * sizeof(size_t));

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(h_x, 0.0, 100.0);
        correct.fill(0);

        COPY_H2D(d_x, h_x.data(), TEST_SIZE * sizeof(double));
        COPY_H2D(d_bins, correct.data(), 10 * sizeof(size_t));

        // compute correct result
        correctBinsBy10Count(h_x, correct);

        // compute test result
        binsBy10Count<<<gridSize, blockSize>>>(d_x, TEST_SIZE, d_bins);
        SYNC();

        // copy back
        COPY_D2H(test.data(), d_bins, 10 * sizeof(size_t));
        
        if (!std::equal(correct.begin(), correct.end(), test.begin())) {
            FREE(d_x);
            FREE(d_bins);
            return false;
        }
    }

    FREE(d_x);
    FREE(d_bins);
    return true;
}

void destroy(Context *ctx) {
    FREE(ctx->d_x);
    FREE(ctx->d_bins);
    delete ctx;
}
