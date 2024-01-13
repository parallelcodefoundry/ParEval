// Driver for 06_fft_dft for CUDA and HIP
// /* Compute the discrete fourier transform of x. Store the result in output.
//    Use CUDA to compute in parallel. The kernel is launched with at least N threads.
//    Example:
// 
//    input: [1, 4, 9, 16]
//    output: [30+0i, -8-12i, -10-0i, -8+12i]
// */
// __global__ void dft(const double *x, cuDoubleComplex *output, size_t N) {

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
    DOUBLE_COMPLEX_T *d_output;
    std::vector<double> h_x;
    std::vector<std::complex<double>> h_output;
    size_t N;
    dim3 blockSize, gridSize;
};

void reset(Context *ctx) {
    fillRand(h_x, -1.0, 1.0);
    COPY_H2D(ctx->d_x, ctx->h_x.data(), ctx->N * sizeof(double));
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = 1 << 18;
    ctx->blockSize = dim3(1024);
    ctx->gridSize = dim3((ctx->N + ctx->blockSize.x - 1) / ctx->blockSize.x); // at least enough threads

    ctx->h_x.resize(ctx->N);
    ctx->h_output.resize(ctx->N);

    ALLOC(ctx->d_x, ctx->N * sizeof(double));
    ALLOC(ctx->d_output, ctx->N * sizeof(DOUBLE_COMPLEX_T));

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    dft<<<ctx->gridSize, ctx->blockSize>>>(ctx->d_x, ctx->d_output, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctDft(ctx->h_x, ctx->h_output);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    dim3 blockSize = dim3(1024);
    dim3 gridSize = dim3((TEST_SIZE + blockSize.x - 1) / blockSize.x); // at least enough threads

    std::vector<double> h_x(TEST_SIZE);
    std::vector<std::complex<double>> correct(TEST_SIZE);

    double *d_x;
    DOUBLE_COMPLEX_T *d_output;
    std::vector<DOUBLE_COMPLEX_T> test(TEST_SIZE);
    ALLOC(d_x, TEST_SIZE * sizeof(double));
    ALLOC(d_output, TEST_SIZE * sizeof(DOUBLE_COMPLEX_T));

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(h_x, -1.0, 1.0);
        COPY_H2D(d_x, h_x.data(), TEST_SIZE * sizeof(double));

        // compute correct result
        correctDft(h_x, correct);

        // compute test result
        dft<<<gridSize, blockSize>>>(d_x, d_output, TEST_SIZE);
        SYNC();

        // copy result back
        COPY_D2H(test.data(), d_output, TEST_SIZE * sizeof(DOUBLE_COMPLEX_T));
        
        for (int j = 0; j < TEST_SIZE; j += 1) {
            if (std::abs(test[j].x - correct[j].real()) > 1e-4 || std::abs(test[j].y - correct[j].imag()) > 1e-4) {
                FREE(d_x);
                FREE(d_output);
                return false;
            }
        }
    }
    FREE(d_x);
    FREE(d_output);
    return true;
}

void destroy(Context *ctx) {
    FREE(ctx->d_x);
    FREE(ctx->d_output);
    delete ctx;
}
