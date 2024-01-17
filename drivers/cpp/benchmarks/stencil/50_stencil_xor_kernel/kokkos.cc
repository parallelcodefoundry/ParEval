// Driver for 50_stencil_xor_kernel for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Set every cell's value to 1 if it has exactly one neighbor that's a 1. Otherwise set it to 0.
//    Note that we only consider neighbors and not input_{i,j} when computing output_{i,j}.
//    input and output are NxN grids of ints.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Example:
// 
//    input: [[0, 1, 1, 0],
//            [1, 0, 0, 0],
//            [0, 0, 0, 0],
//            [0, 1, 0, 0]
//    output: [[0, 0, 1, 1],
//             [1, 0, 0, 1],
//             [0, 0, 1, 0],
//             [1, 0, 1, 0]]
// */
// void cellsXOR(Kokkos::View<const int**> &input, Kokkos::View<int**> &output, size_t N) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "kokkos-includes.hpp"
#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<int**> input;
    Kokkos::View<int**> output;
    size_t N;

    std::vector<int> input_host;
    std::vector<int> output_host;
};

void reset(Context *ctx) {
    fillRand(ctx->input_host, 0, 1);
    std::fill(ctx->output_host.begin(), ctx->output_host.end(), 0);

    copyVectorToView(ctx->input_host, ctx->input);
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), ctx->output, 0);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    ctx->input_host.resize(ctx->N * ctx->N);
    ctx->output_host.resize(ctx->N * ctx->N);

    ctx->input = Kokkos::View<int**>("input", ctx->N, ctx->N);
    ctx->output = Kokkos::View<int**>("output", ctx->N, ctx->N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    cellsXOR(ctx->input, ctx->output, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctCellsXOR(ctx->input_host, ctx->output_host, ctx->N);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<int> input_host(TEST_SIZE * TEST_SIZE), correct(TEST_SIZE * TEST_SIZE);

    Kokkos::View<int**> input("input", TEST_SIZE, TEST_SIZE);
    Kokkos::View<int**> test("test", TEST_SIZE, TEST_SIZE);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(input_host, 0, 1);
        std::fill(correct.begin(), correct.end(), 0);

        copyVectorToView(input_host, input);
        Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), test, 0);

        // compute correct result
        correctCellsXOR(input_host, correct, TEST_SIZE);

        // compute test result
        cellsXOR(input, test, TEST_SIZE);

        for (int i = 0; i < correct.size(); i += 1) {
            if (correct[i] != test[i]) {
                return false;
            }
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
