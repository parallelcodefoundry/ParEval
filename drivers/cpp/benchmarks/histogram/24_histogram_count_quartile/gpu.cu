// Driver for 24_histogram_count_quartile for CUDA and HIP
// /* Count the number of doubles in the vector x that have a fractional part 
//    in [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1). Store the counts in `bins`.
//    Use CUDA to compute in parallel. The kernel is launched with at least N threads.
//    Examples:
// 
//    input: [7.8, 4.2, 9.1, 7.6, 0.27, 1.5, 3.8]
//    output: [2, 1, 2, 2]
// 
//    input: [1.9, 0.2, 0.6, 10.1, 7.4]
//    output: [2, 1, 1, 1]
// */
// __global__ void countQuartiles(const double *x, size_t N, size_t bins[4]) {

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
    std::array<size_t, 4> h_bins;
    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, 0.0, 100.0);
    ctx->h_bins.fill(0);

    COPY_H2D(ctx->d_x, ctx->h_x.data(), ctx->N * sizeof(double));
    COPY_H2D(ctx->d_bins, ctx->h_bins.data(), 4 * sizeof(size_t));
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = 1 << 18;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    ctx->h_x.resize(ctx->N);
    ALLOC(ctx->d_x, ctx->N * sizeof(double));
    ALLOC(ctx->d_bins, 4 * sizeof(size_t));

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    countQuartiles<<<ctx->gridSize, ctx->blockSize>>>(ctx->d_x, ctx->N, ctx->d_bins);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctCountQuartiles(ctx->h_x, ctx->h_bins);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    std::vector<double> h_x(TEST_SIZE);
    std::array<size_t, 4> correct, test;

    double *d_x;
    size_t *d_bins;
    ALLOC(d_x, TEST_SIZE * sizeof(double));
    ALLOC(d_bins, 4 * sizeof(size_t));

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(h_x, 0.0, 100.0);
        correct.fill(0);

        COPY_H2D(d_x, h_x.data(), TEST_SIZE * sizeof(double));
        COPY_H2D(d_bins, correct.data(), 4 * sizeof(size_t));

        // compute correct result
        correctCountQuartiles(h_x, correct);

        // compute test result
        countQuartiles<<<gridSize, blockSize>>>(d_x, TEST_SIZE, d_bins);
        SYNC();
        
        // copy back
        COPY_D2H(test.data(), d_bins, 4 * sizeof(size_t));
        
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
