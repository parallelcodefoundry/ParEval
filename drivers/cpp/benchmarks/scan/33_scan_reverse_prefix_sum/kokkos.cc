// Driver for 33_scan_reverse_prefix_sum for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Compute the reverse prefix sum of the array x into output.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Examples:
//    
//    input: [1, 7, 4, 6, 6, 2]
//    output: [2, 8, 14, 18, 25, 26]
// 
//    input: [3, 3, 7, 1, -2]
//    output: [-2, -1, 6, 9, 12]
// */
// void reversePrefixSum(Kokkos::View<const int*> const& x, Kokkos::View<int*> &output) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<const int*> x;
    Kokkos::View<int*> xNonConst;
    Kokkos::View<int*> output;

    std::vector<int> h_x;
    std::vector<int> h_output;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, -100, 100);
    std::fill(ctx->h_output.begin(), ctx->h_output.end(), 0);

    copyVectorToView(ctx->h_x, ctx->xNonConst);
    ctx->x = ctx->xNonConst;
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), ctx->output, 0);
}

Context *init() {
    Context *ctx = new Context();

    ctx->h_x.resize(1 << 18);

    ctx->xNonConst = Kokkos::View<int*>("x", 1 << 18);
    ctx->output = Kokkos::View<int*>("output", 1 << 18);

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    reversePrefixSum(ctx->x, ctx->output);
}

void best(Context *ctx) {
    correctReversePrefixSum(ctx->h_x, ctx->h_output);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<int> h_x(TEST_SIZE), correct(TEST_SIZE);

    Kokkos::View<int*> xNonConst("x", TEST_SIZE);
    Kokkos::View<const int*> x;
    Kokkos::View<int*> test("test");

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(h_x, -100, 100);
        std::fill(correct.begin(), correct.end(), 0);

        copyVectorToView(h_x, xNonConst);
        x = xNonConst;
        Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), test, 0);

        // compute correct result
        correctReversePrefixSum(h_x, correct);

        // compute test result
        reversePrefixSum(x, test);

        for (int i = 0; i < TEST_SIZE; i++) {
            if (correct[i] != test[i]) {
                return false;
            }
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
