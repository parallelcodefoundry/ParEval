// Driver for 41_sort_k-th_smallest_element for CUDA and HIP
// /* Find the k-th smallest element of the vector x.
//    Use CUDA to compute in parallel. The kernel is launched with at least as many threads as values in x.
//    Example:
//    
//    input: x=[1, 7, 6, 0, 2, 2, 10, 6], k=4
//    output: 6
// */
// __global__ void findKthSmallest(const int *x, size_t N, int k, int *kthSmallest) {

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
    int *d_x, *d_kthSmallest;
    std::vector<int> h_x;
    int k;
    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, 0, 10000);
    ctx->k = rand() % ctx->N;

    COPY_H2D(ctx->d_x, ctx->h_x.data(), ctx->N * sizeof(int));
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = 1 << 15;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    ctx->h_x.resize(ctx->N);
    ALLOC(ctx->d_x, ctx->N * sizeof(int));
    ALLOC(ctx->d_kthSmallest, sizeof(int));

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    findKthSmallest<<<ctx->gridSize, ctx->blockSize>>>(ctx->d_x, ctx->N, ctx->k, ctx->d_kthSmallest);
}

void best(Context *ctx) {
    int sm = correctFindKthSmallest(ctx->h_x, ctx->k);
    (void)sm;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    std::vector<int> h_x(TEST_SIZE);
    int k;
    int *d_x, *d_kthSmallest;
    ALLOC(d_x, TEST_SIZE * sizeof(int));
    ALLOC(d_kthSmallest, sizeof(int));

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(h_x, 0, 10000);
        k = rand() % TEST_SIZE;

        COPY_H2D(d_x, h_x.data(), TEST_SIZE * sizeof(int));

        // compute correct result
        int correct = correctFindKthSmallest(h_x, k);

        // compute test result
        findKthSmallest<<<gridSize, blockSize>>>(d_x, TEST_SIZE, k, d_kthSmallest);
        SYNC();

        // copy back
        int test;
        COPY_D2H(&test, d_kthSmallest, sizeof(int));

        if (test != correct) {
            FREE(d_x);
            FREE(d_kthSmallest);
            return false;
        }
    }

    FREE(d_x);
    FREE(d_kthSmallest);
    return true;
}

void destroy(Context *ctx) {
    FREE(ctx->d_x);
    FREE(ctx->d_kthSmallest);
    delete ctx;
}