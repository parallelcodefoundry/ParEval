// Driver for 38_search_find_the_first_even_number for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Return the index of the first even number in the vector x.
//    Examples:
// 
//    input: [7, 3, 9, 5, 5, 7, 2, 9, 12, 11]
//    output: 6
// 
//    input: [3, 8, 9, 9, 3, 4, 8, 6]
//    output: 1
// */
// size_t findFirstEven(std::vector<int> const& x) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    std::vector<int> x;
};

void reset(Context *ctx) {
    fillRand(ctx->x, 1, 20);
    for (int i = 0; i < ctx->x.size(); i += 1) {
        ctx->x[i] = 2 * ctx->x[i] + 1;  // make everything odd
    }
    // make two values in the middle quadrants even
    size_t min = ctx->x.size() / 4;
    size_t max = 3 * ctx->x.size() / 4;
    ctx->x[rand() % (max - min) + min] += 1;
    ctx->x[rand() % (max - min) + min] += 1;
    BCAST(ctx->x, INT);
}

Context *init() {
    Context *ctx = new Context();

    ctx->x.resize(1 << 18);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    size_t idx = findFirstEven(ctx->x);
    (void)idx;
}

void NO_OPTIMIZE best(Context *ctx) {
    size_t idx = correctFindFirstEven(ctx->x);
    (void)idx;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        std::vector<int> x(TEST_SIZE);
        fillRand(x, 1, 100);
        BCAST(x, INT);

        // compute correct result
        size_t correct = correctFindFirstEven(x);

        // compute test result
        size_t test = findFirstEven(x);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && correct != test) {
            isCorrect = false;
        }
        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
