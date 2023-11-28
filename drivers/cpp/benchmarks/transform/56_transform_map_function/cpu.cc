// Driver for 56_transform_map_function
// bool isPowerOfTwo(int x) {
//     return (x > 0) && !(x & (x - 1));
// }
// 
// /* Apply the isPowerOfTwo function to every value in x and store the results in mask.
//    Example:
//    
//    input: [8, 0, 9, 7, 15, 64, 3]
//    output: [true, false, false, false, false, true, false]
// */
// void mapPowersOfTwo(std::vector<int> const& x, std::vector<bool> &mask) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM


struct Context {
    std::vector<int> x;
    std::vector<bool> mask;
};

void reset(Context *ctx) {
    fillRand(ctx->x, 1, 1025);
    BCAST(ctx->x, INT);
}

Context *init() {
    Context *ctx = new Context();
    ctx->x.resize(100000);
    ctx->mask.resize(100000);
    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    mapPowersOfTwo(ctx->x, ctx->mask);
}

void best(Context *ctx) {
    correctMapPowersOfTwo(ctx->x, ctx->mask);
}

bool validate(Context *ctx) {

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<int> input(1024);
        fillRand(input, 1, 1025);
        BCAST(input, INT);

        // compute correct result
        std::vector<bool> correctResult(input.size());
        correctMapPowersOfTwo(input, correctResult);

        // compute test result
        std::vector<bool> testResult(input.size());
        mapPowersOfTwo(input, testResult);
        SYNC();
        
        if (IS_ROOT(rank) && !std::equal(correctResult.begin(), correctResult.end(), testResult.begin())) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}


