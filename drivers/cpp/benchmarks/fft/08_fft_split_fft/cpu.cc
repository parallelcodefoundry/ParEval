// Driver for 08_fft_split_fft for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Compute the fourier transform of x. Store real part of results in r and imaginary in i.
//    Example:
// 
//    input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
//    output: r: [4, 1, 0, 1, 0, 1, 0, 1] i: [0, -2.41421, 0, -0.414214, 0, 0.414214, 0, 2.41421]
// */
// void fft(std::vector<std::complex<double>> const& x, std::vector<double> &r, std::vector<double> &i) {

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    std::vector<std::complex<double>> x;
    std::vector<double> real, imag;
};

void reset(Context *ctx) {
    fillRand(ctx->real, -1.0, 1.0);
    fillRand(ctx->imag, -1.0, 1.0);
    BCAST(ctx->real, DOUBLE);
    BCAST(ctx->imag, DOUBLE);
    
    for (size_t i = 0; i < ctx->x.size(); i += 1) {
        ctx->x[i] = std::complex<double>(ctx->real[i], ctx->imag[i]);
    }
}

Context *init() {
    Context *ctx = new Context();

    ctx->x.resize(1 << 18);
    ctx->real.resize(1 << 18);
    ctx->imag.resize(1 << 18);

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    fft(ctx->x, ctx->real, ctx->imag);
}

void best(Context *ctx) {
    correctFft(ctx->x, ctx->real, ctx->imag);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<double> real(TEST_SIZE), imag(TEST_SIZE);
    std::vector<double> correctReal(TEST_SIZE), correctImag(TEST_SIZE);
    std::vector<double> testReal(TEST_SIZE), testImag(TEST_SIZE);
    std::vector<std::complex<double>> x(TEST_SIZE);

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(real, -1.0, 1.0);
        fillRand(imag, -1.0, 1.0);
        BCAST(real, DOUBLE);
        BCAST(imag, DOUBLE);

        for (size_t j = 0; j < x.size(); j += 1) {
            x[j] = std::complex<double>(real[j], imag[j]);
        }

        // compute correct result
        correctFft(x, correctReal, correctImag);

        // compute test result
        fft(x, testReal, testImag);
        SYNC();
        
        if (IS_ROOT(rank) && (!fequal(correctReal, testReal, 1e-4) || !fequal(correctImag, testImag, 1e-4))) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
