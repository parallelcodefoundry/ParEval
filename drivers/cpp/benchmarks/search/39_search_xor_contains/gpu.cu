// Driver for 39_search_xor_contains for CUDA and HIP
// /* Set `found` to true if `val` is only in one of vectors x or y.
//    Set it to false if it is in both or neither.
//    Use CUDA to search in parallel. The kernel is launched with at least N threads.
//    Examples:
// 
//    input: x=[1,8,4,3,2], y=[3,4,4,1,1,7], val=7
//    output: true
// 
//    input: x=[1,8,4,3,2], y=[3,4,4,1,1,7], val=1
//    output: false
// */
// __global__ void xorContains(const int *x, const int *y, size_t N, int val, bool *found) {

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
    int *d_x, *d_y;
    bool *d_found;
    std::vector<int> h_x, h_y;
    int val;
    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, -100, 100);
    fillRand(ctx->h_y, -100, 100);
    ctx->val = rand() % 200 - 100;

    COPY_H2D(ctx->d_x, ctx->h_x.data(), ctx->N * sizeof(int));
    COPY_H2D(ctx->d_y, ctx->h_y.data(), ctx->N * sizeof(int));
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = 1 << 18;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    ctx->h_x.resize(ctx->N);
    ctx->h_y.resize(ctx->N);
    ALLOC(ctx->d_x, ctx->N * sizeof(int));
    ALLOC(ctx->d_y, ctx->N * sizeof(int));
    ALLOC(ctx->d_found, 1 * sizeof(bool));

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    xorContains<<<ctx->gridSize, ctx->blockSize>>>(ctx->d_x, ctx->d_y, ctx->N, ctx->val, ctx->d_found);
}

void best(Context *ctx) {
    bool found = correctXorContains(ctx->h_x, ctx->h_y, ctx->val);
    (void)found;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        std::vector<int> h_x(TEST_SIZE);
        std::vector<int> h_y(TEST_SIZE);
        fillRand(h_x, -100, 100);
        fillRand(h_y, -100, 100);
        int val = rand() % 200 - 100;

        int *d_x, *d_y;
        bool *d_found;
        ALLOC(d_x, TEST_SIZE * sizeof(int));
        ALLOC(d_y, TEST_SIZE * sizeof(int));
        ALLOC(d_found, 1 * sizeof(bool));
        COPY_H2D(d_x, h_x.data(), TEST_SIZE * sizeof(int));
        COPY_H2D(d_y, h_y.data(), TEST_SIZE * sizeof(int));

        // compute correct result
        bool found = correctXorContains(h_x, h_y, val);

        // compute test result
        xorContains<<<gridSize, blockSize>>>(d_x, d_y, TEST_SIZE, val, d_found);
        SYNC();

        // check result
        bool testFound;
        COPY_D2H(&testFound, d_found, 1 * sizeof(bool));
        
        if (testFound != found) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
