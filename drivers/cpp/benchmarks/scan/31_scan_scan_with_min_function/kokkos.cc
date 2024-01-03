// /* Replace the i-th element of the array x with the minimum value from indices 0 through i.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Examples:
//
//    input: [8, 6, -1, 7, 3, 4, 4]
//    output: [8, 6, -1, -1, -1, -1, -1]
//
//    input: [5, 4, 6, 4, 3, 6, 1, 1]
//    output: [5, 4, 4, 4, 3, 3, 1, 1]
// */
// void partialMinimums(Kokkos::View<float*> &x) {

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
    Kokkos::View<float*> x;
    std::vector<float> x_host;
};

void reset(Context *ctx) {
    fillRand(ctx->x_host, -100.0, 100.0);
    copyVectorToView(ctx->x_host, ctx->x);
}

Context *init() {
    Context *ctx = new Context();
    ctx->x_host.resize(100000);
    ctx->x = Kokkos::View<float*>("x", 100000);
    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    partialMinimums(ctx->x);
}

void best(Context *ctx) {
    correctPartialMinimums(ctx->x_host);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<float> x_host(TEST_SIZE), test(TEST_SIZE);
    Kokkos::View<float*> x("x", TEST_SIZE);

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x_host, -100.0, 100.0);
        copyVectorToView(x_host, x);

        // compute correct result
        correctPartialMinimums(x_host);

        // compute test result
        partialMinimums(x);

        // copy to vector
        copyViewToVector(x, test);

        if (!std::equal(x_host.begin(), x_host.end(), test.begin())) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
