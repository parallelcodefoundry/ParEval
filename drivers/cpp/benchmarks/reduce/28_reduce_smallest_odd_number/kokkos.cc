// Driver for 28_reduce_smallest_odd_number for Kokkos
// #include <Kokkos_Core.hpp>
//
// /* Return the value of the smallest odd number in the vector x.
//    Use Kokkos to compute in parallel. Assume Kokkos is already initialized.
//    Examples:
//
//    input: [7, 9, 5, 2, 8, 16, 4, 1]
//    output: 1
//
//    input: [8, 36, 7, 2, 11]
//    output: 7
// */
// int smallestOdd(Kokkos::View<const int*> const& x) {

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
    Kokkos::View<const int*> x;
    Kokkos::View<int*> xNonConst;

    std::vector<int> x_host;
    int val;
};

void reset(Context *ctx) {
    fillRand(ctx->x_host, 0.0, 100.0);

    copyVectorToView(ctx->x_host, ctx->xNonConst);
    ctx->x = ctx->xNonConst;
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(1 << 18);

    ctx->xNonConst = Kokkos::View<int*>("x", 1 << 18);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    ctx->val = smallestOdd(ctx->x);
    (void) ctx->val;
}

void best(Context *ctx) {
    ctx->val = correctSmallestOdd(ctx->x_host);
    (void) ctx->val;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<int> x_host(TEST_SIZE);
    int test, correct;

    Kokkos::View<int*> xNonConst("x", TEST_SIZE);
    Kokkos::View<const int*> x;

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x_host, 0.0, 100.0);

        copyVectorToView(x_host, xNonConst);
        x = xNonConst;

        // compute correct result
        correct = correctSmallestOdd(x_host);

        // compute test result
        test = smallestOdd(x);

        if (correct != test) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
