// Driver for 40_sort_sort_an_array_of_complex_numbers_by_magnitude for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Sort the array x of complex numbers by their magnitude in ascending order.
//    Use Kokkos to sort in parallel. Assume Kokkos has already been initialized.
//    Example:
//    
//    input: [3.0-1.0i, 4.5+2.1i, 0.0-1.0i, 1.0-0.0i, 0.5+0.5i]
//    output: [0.5+0.5i, 0.0-1.0i, 1.0-0.0i, 3.0-1.0i, 4.5+2.1i]
// */
// void sortComplexByMagnitude(Kokkos::View<Kokkos::complex<double>*> &x) {

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
    Kokkos::View<Kokkos::complex<double>*> x;
    std::vector<std::complex<double>> x_host;
};

void reset(Context *ctx) {
    for (int i = 0; i < ctx->x_host.size(); i += 1) {
        const double real = (rand() / (double) RAND_MAX) * 100.0;
        const double imag = (rand() / (double) RAND_MAX) * 100.0;
        ctx->x_host[i] = std::complex<double>(real, imag);
        ctx->x(i) = Kokkos::complex<double>(real, imag);
    }
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(1 << 15);
    ctx->x = Kokkos::View<Kokkos::complex<double>*>("x", ctx->x_host.size());

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    sortComplexByMagnitude(ctx->x);
}

void best(Context *ctx) {
    correctSortComplexByMagnitude(ctx->x_host);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<std::complex<double>> correct(TEST_SIZE);
    Kokkos::View<Kokkos::complex<double>*> test("test", TEST_SIZE);

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        for (int i = 0; i < correct.size(); i += 1) {
            const double real = (rand() / (double) RAND_MAX) * 100.0;
            const double imag = (rand() / (double) RAND_MAX) * 100.0;
            correct[i] = std::complex<double>(real, imag);
            test(i) = Kokkos::complex<double>(real, imag);
        }

        // compute correct result
        correctSortComplexByMagnitude(correct);

        // compute test result
        sortComplexByMagnitude(test);
        
        for (int i = 0; i < correct.size(); i += 1) {
            if (correct[i] != test(i)) {
                return false;
            }
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
