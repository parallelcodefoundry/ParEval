// Driver for 41_sort_k-th_smallest_element for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Find the k-th smallest element of the vector x.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Example:
//    
//    input: x=[1, 7, 6, 0, 2, 2, 10, 6], k=4
//    output: 6
// */
// int findKthSmallest(Kokkos::View<const int*> const& x, int k) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "kokkos-includes.hpp"

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<int*> x;
    int k;
    std::vector<int> x_host;
};

void reset(Context *ctx) {
    fillRand(ctx->x_host, 0, 10000);
    k = rand() % ctx->x_host.size();
    copyVectorToView(ctx->x_host, ctx->x);
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(DRIVER_PROBLEM_SIZE);
    ctx->x = Kokkos::View<int*>("x", ctx->x_host.size());

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    int sm = findKthSmallest(ctx->x, ctx->k);
    (void)sm;
}

void NO_OPTIMIZE best(Context *ctx) {
    int sm = correctFindKthSmallest(ctx->x_host, ctx->k);
    (void)sm;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<int> x_host(TEST_SIZE);
    int k;
    Kokkos::View<int*> x("x", TEST_SIZE);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x_host, 0, 10000);
        k = rand() % TEST_SIZE;
        copyVectorToView(x_host, x);

        // compute correct result
        int correct = correctFindKthSmallest(x_host, k);

        // compute test result
        int test = findKthSmallest(x, k);
        
        if (test != correct) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
