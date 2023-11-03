// Driver for 0_sort_non-zero
// /* Sort the vector x in ascending order ignoring elements with value 0.
//    Leave zero valued elements in-place.
//    Example:
// 
// 	  input: [8, 4, 0, 9, 8, 0, 1, -1, 7]
//    output: [-1, 1, 0, 4, 7, 0, 8, 8, 9]
// */
// void sortIgnoreZero(int *x, size_t N) {

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
};

void fillRandWithZeroes(std::vector<int> &x) {
    // fill x with random values, but set some to zero
    for (int i = 0; i < x.size(); i += 1) {
        x[i] = rand();
        if (rand() % 5) {
            x[i] = 0;
        }
    }
}

void reset(Context *ctx) {
    fillRandWithZeroes(ctx->cpuScratch);
    COPY_H2D(ctx->x, ctx->cpuScratch.data(), ctx->N * sizeof(int));
}

Context *init() {
    Context *ctx = new Context();
    ctx->N = 10000;
    ALLOC(ctx->x, ctx->N * sizeof(int));
    ctx->cpuScratch.resize(ctx->N);
    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    sortIgnoreZero<<<ctx->N,1,0>>>(ctx->x, ctx->N);
}

void best(Context *ctx) {
    correctSortIgnoreZero(ctx->cpuScratch);
}

bool validate(Context *ctx) {

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<int> input(1024);
        fillRandWithZeroes(input);

        // compute correct result
        std::vector<int> correctResult = input;
        correctSortIgnoreZero(correctResult);

        // compute test result
        int *testResultDevice;
        ALLOC(testResultDevice, input.size() * sizeof(int));
        COPY_H2D(testResultDevice, input.data(), input.size() * sizeof(int));
        sortIgnoreZero<<<ctx->N,1,0>>>(testResultDevice, ctx->N);
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
    delete ctx;
}


