// Driver for 34_scan_largest_contiguous_subarray_sum for CUDA and HIP
// /* Compute the largest sum of any contiguous subarray in the vector x.
//    i.e. if x=[−2, 1, −3, 4, −1, 2, 1, −5, 4] then [4, −1, 2, 1] is the contiguous
//    subarray with the largest sum of 6.
//    Store the result in sum.
//    Use CUDA to compute in parallel. The kernel is launched with at least as many threads as values in x.
//    Example:
//
//    input: [−2, 1, −3, 4, −1, 2, 1, −5, 4]
//    output: 6
// */
// __global__ void maximumSubarray(const int *x, size_t N, int *sum) {

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
    int *d_x;
    int *d_sum;
    std::vector<int> h_x;
    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, -100, 100);

    COPY_H2D(ctx->d_x, ctx->h_x.data(), ctx->N * sizeof(int));
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = 100000;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    ctx->h_x.resize(ctx->N);
    ALLOC(ctx->d_x, ctx->N * sizeof(int));
    ALLOC(ctx->d_sum, sizeof(int));

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    maximumSubarray<<<ctx->gridSize, ctx->blockSize>>>(ctx->d_x, ctx->N, ctx->d_sum);
}

void NO_OPTIMIZE best(Context *ctx) {
    int val = correctMaximumSubarray(ctx->h_x);
    (void) val;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    std::vector<int> h_x(TEST_SIZE);
    int correct, test;

    int *d_x, *d_test;
    ALLOC(d_x, TEST_SIZE * sizeof(int));
    ALLOC(d_test, sizeof(int));

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(h_x, -100, 100);
        correct = 0;
        test = 0;

        COPY_H2D(d_x, h_x.data(), TEST_SIZE * sizeof(int));
        COPY_H2D(d_test, &test, sizeof(int));

        // compute correct result
        correct = correctMaximumSubarray(h_x);

        // compute test result
        maximumSubarray<<<gridSize, blockSize>>>(d_x, TEST_SIZE, d_test);
        SYNC();

        // copy back
        COPY_D2H(&test, d_test, sizeof(int));

        if (test != correct) {
            FREE(d_x);
            FREE(d_test);
            return false;
        }
    }

    FREE(d_x);
    FREE(d_test);
    return true;
}

void destroy(Context *ctx) {
    FREE(ctx->d_x);
    FREE(ctx->d_sum);
    delete ctx;
}
