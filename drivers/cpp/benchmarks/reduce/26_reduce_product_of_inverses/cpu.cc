// Driver for 26_reduce_product_of_inverses for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Return the product of the vector x with every odd indexed element inverted.
//    i.e. x_0 * 1/x_1 * x_2 * 1/x_3 * x_4 ...
//    Example:
//
//    input: [4, 2, 10, 4, 5]
//    output: 25
// */
// double productWithInverses(std::vector<double> const& x) {

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

void compute(Context *ctx) {
    ctx->val = productWithInverses(ctx->x);
    (void) ctx->val;
}

void best(Context *ctx) {
    ctx->val = correctProductWithInverses(ctx->x);
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
        correct = correctProductWithInverses(x);

        // compute test result
        test = productWithInverses(x);
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