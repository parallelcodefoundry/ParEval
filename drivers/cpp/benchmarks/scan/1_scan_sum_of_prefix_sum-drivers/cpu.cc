// Driver for 1_scan_sum_of_prefix_sum
// /* Compute the prefix sum array of the vector x and return its sum.
//    Example:
// 
//    input: [-7, 2, 1, 9, 4, 8]
//    output: 15
// */
// double sumOfPrefixSum(std::vector<double> const& x) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM


struct Context {
    std::vector<double> x;
};

void reset(Context *ctx) {
    fillRand(ctx->x, -100.0, 100.0);
}

Context *init() {
    Context *ctx = new Context();
    ctx->x.resize(1 << 20);
    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    double val = sumOfPrefixSum(ctx->x);
    (void) val;
}

void best(Context *ctx) {
    double val = correctSumOfPrefixSum(ctx->x);
    (void) val;
}

bool validate(Context *ctx) {

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<double> input(2048);
        fillRand(input, -100.0, 100.0);

        // compute correct result
        double correctResult = correctSumOfPrefixSum(input);

        // compute test result
        double testResult = sumOfPrefixSum(input);
        
        if (std::fabs(correctResult - testResult) > 1e-6) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}

