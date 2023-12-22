// Driver for 21_histogram_bin_0-100 for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Vector x contains values between 0 and 100, inclusive. Count the number of
//    values in [0,10), [10, 20), [20, 30), ... and store the counts in `bins`.
//    Example:
// 
//    input: [7, 32, 95, 12, 39, 32, 11, 71, 70, 66]
//    output: [1, 2, 0, 3, 0, 0, 1, 2, 0, 1]
// */
// void binsBy10Count(std::vector<double> const& x, std::array<size_t, 10> &bins) {

#include <array>
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
    std::array<size_t, 10> bins;
};

void reset(Context *ctx) {
    fillRand(ctx->x, 0.0, 100.0);
    BCAST(ctx->x, DOUBLE);
    ctx->bins.fill(0);
}

Context *init() {
    Context *ctx = new Context();

    ctx->x.resize(1 << 18);

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    binsBy10Count(ctx->x, ctx->bins);
}

void best(Context *ctx) {
    correctBinsBy10Count(ctx->x, ctx->bins);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<double> x(TEST_SIZE);
    std::array<size_t, 10> correct, test;

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(x, 0.0, 100.0);
        BCAST(x, DOUBLE);
        bins.fill(0);

        // compute correct result
        correctBinsBy10Count(x, correct);

        // compute test result
        binsBy10Count(x, test);
        SYNC();
        
        if (IS_ROOT(rank) && !std::equal(correct.begin(), correct.end(), test.begin())) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}