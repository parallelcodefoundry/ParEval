// Driver for 27_reduce_average for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Return the average of the vector x.
//    Examples:
//
//    input: [1, 8, 4, 5, 1]
//    output: 3.8
//
//    input: [2, 2, 2, 3]
//    output: 2.25
// */
// double average(std::vector<double> const& x) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    std::vector<double> x;
    double val;
};

void reset(Context *ctx) {
    fillRand(ctx->x, 0.0, 100.0);
    BCAST(ctx->x, DOUBLE);
}

Context *init() {
    Context *ctx = new Context();

    ctx->x.resize(1 << 18);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    ctx->val = average(ctx->x);
    (void) ctx->val;
}

void best(Context *ctx) {
    ctx->val = correctAverage(ctx->x);
    (void) ctx->val;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<double> x(TEST_SIZE);
    double test, correct;

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x, 0.0, 100.0);
        BCAST(x, DOUBLE);

        // compute correct result
        correct = correctAverage(x);

        // compute test result
        test = average(x);
        SYNC();

        if (IS_ROOT(rank) && correct != test) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
