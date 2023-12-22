// Driver for 09_fft_fft_out_of_place for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Compute the discrete fourier transform of x. Store the result in output.
//    Use Kokkos to compute in parallel. Assume Kokkos is already initialized.
//    Example:
// 
//    input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
//    output: [{4,0}, {1,-2.42421}, {0,0}, {1,-0.414214}, {0,0}, {1,0.414214}, {0,0}, {1,2.41421}]
// */
// void fft(Kokkos::View<const Kokkos::complex<double>*> &x, Kokkos::View<Kokkos::complex<double>*> &output) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<Kokkos::complex<double>*> xNonConst, output;
    Kokkos::View<const Kokkos::complex<double>*> x;
    std::vector<std::complex<double>> x_host, output_host;
    std::vector<double> real, imag;
};

void reset(Context *ctx) {
    fillRand(ctx->real, -1.0, 1.0);
    fillRand(ctx->imag, -1.0, 1.0);

    for (size_t i = 0; i < ctx->x_host.size(); i += 1) {
        ctx->x_host[i] = std::complex<double>(ctx->real[i], ctx->imag[i]);
        ctx->xNonConst(i) = Kokkos::complex<double>(ctx->real[i], ctx->imag[i]);
    }
    ctx->x = ctx->xNonConst;
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(1 << 18);
    ctx->output_host.resize(1 << 18);
    ctx->real.resize(1 << 18);
    ctx->imag.resize(1 << 18);
    ctx->xNonConst = Kokkos::View<Kokkos::complex<double>*>("x", 1 << 18);
    ctx->output = Kokkos::View<Kokkos::complex<double>*>("output", 1 << 18);

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    fft(ctx->x, ctx->output);
}

void best(Context *ctx) {
    correctFft(ctx->x_host, ctx->output_host);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<std::complex<double>> x_host(TEST_SIZE), correct(TEST_SIZE);
    std::vector<double> real(TEST_SIZE), imag(TEST_SIZE);
    Kokkos::View<const Kokkos::complex<double>*> x("x", TEST_SIZE);
    Kokkos::View<Kokkos::complex<double>*> xNonConst("output", TEST_SIZE);
    Kokkos::View<Kokkos::complex<double>*> test("test", TEST_SIZE);

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(real, -1.0, 1.0);
        fillRand(imag, -1.0, 1.0);

        for (size_t j = 0; j < x_host.size(); j += 1) {
            x_host[j] = std::complex<double>(real[j], imag[j]);
            xNonConst(j) = Kokkos::complex<double>(real[j], imag[j]);
        }
        x = xNonConst;

        // compute correct result
        correctFft(x_host, correct);

        // compute test result
        fft(x, test);
        
        for (size_t j = 0; j < correct.size(); j += 1) {
            if (std::abs(correct[j].real() - test(j).real()) > 1e-4 || std::abs(correct[j].imag() - test(j).imag()) > 1e-4) {
                return false;
            }
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
