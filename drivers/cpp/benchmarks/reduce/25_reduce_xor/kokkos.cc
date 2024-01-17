// Driver for 25_reduce_xor for Kokkos
// #include <Kokkos_Core.hpp>
//
// /* Return the logical XOR reduction of the vector of bools x.
//    Use Kokkos to reduce in parallel. Assume Kokkos is already initialized.
//    Example:
//
//    input: [false, false, false, true]
//    output: true
// */
// bool reduceLogicalXOR(Kokkos::View<const bool*> const& x) {

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
    Kokkos::View<const bool*> x;
    Kokkos::View<bool*> xNonConst;

    std::vector<bool> x_host;
    bool output, output_host;
};

void reset(Context *ctx) {
    fillRand(ctx->x_host, 0, 1);

    copyVectorToView(ctx->x_host, ctx->xNonConst);
    ctx->x = ctx->xNonConst;
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(1 << 18);

    ctx->xNonConst = Kokkos::View<bool*>("x", 1 << 18);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    ctx->output = reduceLogicalXOR(ctx->x);
}

void best(Context *ctx) {
    ctx->output_host = correctReduceLogicalXOR(ctx->x_host);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<bool> x_host(TEST_SIZE);
    bool correct, test;

    Kokkos::View<bool*> xNonConst("x", TEST_SIZE);
    Kokkos::View<const bool*> x;

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x_host, 0, 1);

        copyVectorToView(x_host, xNonConst);
        x = xNonConst;

        // compute correct result
        correct = correctReduceLogicalXOR(x_host);

        // compute test result
        test = reduceLogicalXOR(x);

        if (correct != test) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
