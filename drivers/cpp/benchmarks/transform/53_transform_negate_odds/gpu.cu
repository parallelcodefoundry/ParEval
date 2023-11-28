// Driver for 53_transform_negate_odds
// /* In the vector x negate the odd values and divide the even values by 2.
//    Use CUDA to compute in parallel. The kernel is launched with at least as many threads as values in x.
//    Example:
//    
//    input: [16, 11, 12, 14, 1, 0, 5]
//    output: [8, -11, 6, 7, -1, 0, -5]
//  */
// __global__ void negateOddsAndHalveEvens(int *x, size_t N) {

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
    int *x;
    size_t N;
    std::vector<int> cpuScratch;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(ctx->cpuScratch, 1, 100);
    COPY_H2D(ctx->x, ctx->cpuScratch.data(), ctx->N * sizeof(int));
}

Context *init() {
    Context *ctx = new Context();
    ctx->N = 100000;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads
    ALLOC(ctx->x, ctx->N * sizeof(int));
    ctx->cpuScratch.resize(ctx->N);
    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    negateOddsAndHalveEvens<<<ctx->blockSize,ctx->gridSize>>>(ctx->x, ctx->N);
}

void best(Context *ctx) {
    correctNegateOddsAndHalveEvens(ctx->cpuScratch);
}

bool validate(Context *ctx) {

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<int> input(1024);
        fillRand(input, 1, 100);

        // compute correct result
        std::vector<int> correctResult = input;
        correctNegateOddsAndHalveEvens(correctResult);

        // compute test result
        int *testResultDevice;
        ALLOC(testResultDevice, input.size() * sizeof(int));
        COPY_H2D(testResultDevice, input.data(), input.size() * sizeof(int));
        dim3 blockSize = dim3(1024);
        dim3 gridSize = dim3((input.size() + blockSize.x - 1) / blockSize.x); // at least enough threads
        negateOddsAndHalveEvens<<<blockSize,gridSize>>>(testResultDevice, input.size());
        SYNC();

        std::vector<int> testResult(input.size());
        COPY_D2H(testResult.data(), testResultDevice, testResult.size() * sizeof(int));
        
        if (!std::equal(correctResult.begin(), correctResult.end(), testResult.begin())) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    FREE(ctx->x);
    delete ctx;
}


