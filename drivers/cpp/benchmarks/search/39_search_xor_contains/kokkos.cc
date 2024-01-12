// Driver for 39_search_xor_contains for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Return true if `val` is only in one of vectors x or y.
//    Return false if it is in both or neither.
//    Use Kokkos to search in parallel. Assume Kokkos has already been initialized.
//    Examples:
// 
//    input: x=[1,8,4,3,2], y=[3,4,4,1,1,7], val=7
//    output: true
// 
//    input: x=[1,8,4,3,2], y=[3,4,4,1,1,7], val=1
//    output: false
// */
// bool xorContains(Kokkos::View<const int*> const& x, Kokkos::View<const int*> const& y, int val) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<int*> xNonConst, yNonConst;
    Kokkos::View<const int*> x, y;
    std::vector<int> x_host, y_host;
    int val;
};

void reset(Context *ctx) {
    fillRand(ctx->x_host, -100, 100);
    fillRand(ctx->y_host, -100, 100);
    ctx->val = rand() % 200 - 100;

    copyVectorToView(ctx->x_host, ctx->xNonConst);
    copyVectorToView(ctx->y_host, ctx->yNonConst);
    ctx->x = ctx->xNonConst;
    ctx->y = ctx->yNonConst;
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(1 << 20);
    ctx->y_host.resize(1 << 20);
    ctx->xNonConst = Kokkos::View<int*>("x", 1 << 20);
    ctx->yNonConst = Kokkos::View<int*>("y", 1 << 20);

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    bool found = xorContains(ctx->x, ctx->y, ctx->val);
    (void)found;
}

void best(Context *ctx) {
    bool found = correctXorContains(ctx->x_host, ctx->y_host, ctx->val);
    (void)found;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        std::vector<int> x_host(TEST_SIZE);
        std::vector<int> y_host(TEST_SIZE);
        fillRand(x_host, -100, 100);
        fillRand(y_host, -100, 100);
        int val = rand() % 200 - 100;

        // set up Kokkos input
        Kokkos::View<int*> xNonConst("x", TEST_SIZE);
        Kokkos::View<int*> yNonConst("y", TEST_SIZE);
        copyVectorToView(x_host, xNonConst);
        copyVectorToView(y_host, yNonConst);
        Kokkos::View<const int*> x = xNonConst;
        Kokkos::View<const int*> y = yNonConst;

        // compute correct result
        bool correct = correctXorContains(x_host, y_host, val);

        // compute test result
        bool test = xorContains(x, y, val);
        
        if (test != correct) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
