// Driver for 27_reduce_average for Kokkos
// #include <Kokkos_Core.hpp>
//
// /* Return the average of the vector x.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Examples:
//
//    input: [1, 8, 4, 5, 1]
//    output: 3.8
//
//    input: [2, 2, 2, 3]
//    output: 2.25
// */
// double average(Kokkos::View<const double*> const& x) {

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
    Kokkos::View<const double*> x;
    Kokkos::View<double*> xNonConst;

    std::vector<double> x_host;
    double val;
};

void reset(Context *ctx) {
    fillRand(ctx->x_host, 0.0, 100.0);

    copyVectorToView(ctx->x_host, ctx->xNonConst);
    ctx->x = ctx->xNonConst;
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(1 << 18);

    ctx->xNonConst = Kokkos::View<double*>("x", 1 << 18);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    ctx->val = average(ctx->x);
    (void) ctx->val;
}

void best(Context *ctx) {
    ctx->val = correctAverage(ctx->x_host);
    (void) ctx->val;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<double> x_host(TEST_SIZE);
    double test, correct;

    Kokkos::View<double*> xNonConst("x", TEST_SIZE);
    Kokkos::View<const double*> x;

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x_host, 0.0, 100.0);

        copyVectorToView(x_host, xNonConst);
        x = xNonConst;

        // compute correct result
        correct = correctAverage(x_host);

        // compute test result
        test = average(x);

        if (correct != test) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
