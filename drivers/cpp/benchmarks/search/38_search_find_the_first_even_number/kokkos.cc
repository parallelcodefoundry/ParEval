// Driver for 38_search_find_the_first_even_number for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Return the index of the first even number in the vector x.
//    Use Kokkos to parallelize the search. Assume Kokkos has already been initialized.
//    Examples:
// 
//    input: [7, 3, 9, 5, 5, 7, 2, 9, 12, 11]
//    output: 6
// 
//    input: [3, 8, 9, 9, 3, 4, 8, 6]
//    output: 1
// */
// size_t findFirstEven(Kokkos::View<const int*> const& x) {

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
};

void reset(Context *ctx) {
    fillRand(ctx->x_host, 1, 100);
    copyVectorToView(ctx->x_host, ctx->xNonConst);
    ctx->x = ctx->xNonConst;
}

Context *init() {
    Context *ctx = new Context();

    ctx->x_host.resize(100000);
    ctx->xNonConst = Kokkos::View<int*>("x", ctx->x_host.size());

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    size_t idx = findFirstEven(ctx->x);
    (void)idx;
}

void best(Context *ctx) {
    size_t idx = correctFindFirstEven(ctx->x_host);
    (void)idx;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        std::vector<int> x(TEST_SIZE);
        fillRand(x, 1, 100);
        Kokkos::View<int*> xNonConst("x", x.size());
        copyVectorToView(x, xNonConst);
        Kokkos::View<const int*> xView = xNonConst;

        // compute correct result
        size_t correct = correctFindFirstEven(x);

        // compute test result
        size_t test = findFirstEven(xView);
        
        if (test != correct) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
