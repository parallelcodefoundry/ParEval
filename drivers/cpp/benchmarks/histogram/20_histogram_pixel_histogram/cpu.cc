// Driver for 20_histogram_pixel_histogram for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Count the number of pixels in image with each grayscale intensity.
//    The vector `image` is a grayscale image with values 0-255.
//    Store the results in `bins`.
//    Example:
//    
//    input: image=[2, 116, 201, 11, 92, 92, 201, 4, 2]
//    output: [0, 0, 2, 0, 1, ...]
// */
//  void pixelCounts(std::vector<int> const& image, std::array<size_t, 256> &bins) {

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
    std::vector<int> image;
    std::array<size_t, 256> bins;
};

void reset(Context *ctx) {
    fillRand(ctx->image, 0, 255);
    BCAST(ctx->image, INT);

    ctx->bins.fill(0);
}

Context *init() {
    Context *ctx = new Context();

    ctx->image.resize(1 << 18);

    reset(ctx);
    return ctx;
}

void compute(Context *ctx) {
    pixelCounts(ctx->image, ctx->bins);
}

void best(Context *ctx) {
    correctPixelCounts(ctx->image, ctx->bins);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<int> image(TEST_SIZE);
    std::array<size_t, 256> correct, test;

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRand(image, 0, 255);
        BCAST(image, INT);
        
        std::fill(correct.begin(), correct.end(), 0);

        // compute correct result
        correctPixelCounts(image, correct);

        // compute test result
        pixelCounts(image, test);
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
