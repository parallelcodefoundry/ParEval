// Driver for 51_stencil_edge_kernel for Serial, OpenMP, MPI, and MPI+OpenMP
// const int edgeKernel[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
// 
// /* Convolve the edge kernel with a grayscale image. Each pixel will be replaced with
//    the dot product of itself and its neighbors with the edge kernel.
//    Use a value of 0 for pixels outside the image's boundaries and clip outputs between 0 and 255.
//    imageIn and imageOut are NxN grayscale images stored in row-major.
//    Store the output of the computation in imageOut.
//    Example:
// 
//    input: [[112, 118, 141, 152],
//            [93, 101, 119, 203],
//            [45, 17, 16, 232],
//            [82, 31, 49, 101]]
//    output: [[255, 255, 255, 255],
//             [255, 147, 0, 255],
//             [36, 0, 0, 255],
//             [255, 39, 0, 255]]
// */
// void convolveKernel(std::vector<int> const& imageIn, std::vector<int> &imageOut, size_t N) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    std::vector<int> input;
    std::vector<int> output;
    size_t N;
};

void reset(Context *ctx) {
    fillRand(ctx->input, 0, 255);
    std::fill(ctx->output.begin(), ctx->output.end(), 0);
    BCAST(ctx->input, INT);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    ctx->input.resize(ctx->N * ctx->N);
    ctx->output.resize(ctx->N * ctx->N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    convolveKernel(ctx->input, ctx->output, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctConvolveKernel(ctx->input, ctx->output, ctx->N);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<int> input(TEST_SIZE * TEST_SIZE), correct(TEST_SIZE * TEST_SIZE), test(TEST_SIZE * TEST_SIZE);

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(input, 0, 255);
        std::fill(test.begin(), test.end(), 0);
        std::fill(correct.begin(), correct.end(), 0);
        BCAST(input, INT);

        // compute correct result
        correctConvolveKernel(input, correct, ctx->N);

        // compute test result
        convolveKernel(input, test, ctx->N);
        SYNC();

        bool isCorrect = true;
        if (IS_ROOT(rank) && !std::equal(correct.begin(), correct.end(), test.begin())) {
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