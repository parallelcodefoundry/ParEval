// Driver for 04_dense_la_gemv for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Multiply the matrix A by the vector x. Store the results in the vector y.
//    A is an MxN matrix, x has N elements, and y has M elements.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Example:
// 
//    input: A=[[1, -1, 2], [0, -3, 1]] x=[2, 1, 0]
//    output: y=[1, -3]
// */
// void gemv(Kokkos::View<const double**> &A, Kokkos::View<const double*> &x, Kokkos::View<double*> &y, size_t M, size_t N) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<double**> A;
    Kokkos::View<double*> x, y;
    std::vector<double> A_host, x_host, y_host;
    size_t M, N;
};

void reset(Context *ctx) {
    fillRand(ctx->A_host, -10.0, 10.0);
    fillRand(ctx->x_host, -10.0, 10.0);

    copyVectorToView(ctx->x_host, ctx->x);

    /* A has to be done manually since it's 2D */
    for (int i = 0; i < ctx->M; i += 1) {
        for (int j = 0; j < ctx->N; j += 1) {
            ctx->A(i, j) = ctx->A_host[i * ctx->N + j];
        }
    }
}

Context *init() {
    Context *ctx = new Context();

    ctx->M = 1 << 11;
    ctx->N = 1 << 11;
    ctx->A_host.resize(ctx->M * ctx->N);
    ctx->x_host.resize(ctx->N);
    ctx->y_host.resize(ctx->M);

    ctx->A = Kokkos::View<double**>("A", ctx->M, ctx->N);
    ctx->x = Kokkos::View<double*>("x", ctx->N);
    ctx->y = Kokkos::View<double*>("y", ctx->M);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    gemv(ctx->A, ctx->x, ctx->y, ctx->M, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctGemv(ctx->A_host, ctx->x_host, ctx->y_host, ctx->M, ctx->N);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<double> A(TEST_SIZE * TEST_SIZE);
    std::vector<double> x(TEST_SIZE);
    std::vector<double> correct(TEST_SIZE), test(TEST_SIZE);

    Kokkos::View<double**> A_view("A", TEST_SIZE, TEST_SIZE);
    Kokkos::View<double*> x_view("x", TEST_SIZE);
    Kokkos::View<double*> y_view("y", TEST_SIZE);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(A, -10.0, 10.0);
        fillRand(x, -10.0, 10.0);

        for (int i = 0; i < TEST_SIZE; i += 1) {
            for (int j = 0; j < TEST_SIZE; j += 1) {
                A_view(i, j) = A[i * TEST_SIZE + j];
            }
        }
        copyVectorToView(x, x_view);

        // compute correct result
        correctGemv(A, x, correct, TEST_SIZE, TEST_SIZE);

        // compute test result
        gemv(A_view, x_view, y_view, TEST_SIZE, TEST_SIZE);
        copyViewToVector(y_view, test);
        
        if (!fequal(correct, test, 1e-4)) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
