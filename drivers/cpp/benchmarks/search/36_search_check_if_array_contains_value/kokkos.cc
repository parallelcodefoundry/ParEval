// Driver for 36_search_check_if_array_contains_value
// #include <Kokkos_Core.hpp>
// 
// /* Return true if the vector x contains the value `target`. Return false otherwise.
//    Use Kokkos to search in parallel. Assume Kokkos has already been initialized.
//    Examples:
// 
//    input: x=[1, 8, 2, 6, 4, 6], target=3
//    output: false
//    
//    input: x=[1, 8, 2, 6, 4, 6], target=8
//    output: true
// */
// bool contains(Kokkos::View<const int*> const& x, int target) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "kokkos-includes.hpp"

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM


struct Context {
    Kokkos::View<const int*> x;
    Kokkos::View<int*> nonConstX;
    std::vector<int> vecX;
    int target;
};

void reset(Context *ctx) {
    fillRandKokkos(ctx->nonConstX, -50, 50);
    ctx->x = ctx->nonConstX;
    copyViewToVector(ctx->nonConstX, ctx->vecX);
    ctx->target = (rand() % 200) - 100;
}

Context *init() {
    Context *ctx = new Context();

    const size_t N = DRIVER_PROBLEM_SIZE;
    ctx->nonConstX = Kokkos::View<int*>("x", N);
    ctx->vecX.resize(N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    bool flag = contains(ctx->nonConstX, ctx->target);
    (void)flag;
}

void NO_OPTIMIZE best(Context *ctx) {
    bool flag = correctContains(ctx->vecX, ctx->target);
    (void)flag;
}

bool validate(Context *ctx) {

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<int> input(1024);
        Kokkos::View<int*> inputView = Kokkos::View<int*>("input", input.size());

        fillRand(input, -50, 50);
        int target = (rand() % 200) - 100;
        copyVectorToView(input, inputView);

        // compute correct result
        bool correctFlag = correctContains(input, target);

        // compute test result
        bool testFlag = contains(inputView, target);
        
        if (correctFlag != testFlag) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}


