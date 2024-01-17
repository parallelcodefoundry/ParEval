// Driver for 48_sparse_la_sparse_axpy for Kokkos
// #include <Kokkos_Core.hpp>
// 
// struct Element {
// 	size_t index;
//   double value;
// };
// 
// /* Compute z = alpha*x+y where x and y are sparse vectors. Store the result in z.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Example:
//    
//    input: x=[{5, 12}, {8, 3}, {12, -1}], y=[{3, 1}, {5, -2}, {7, 1}, {8, -3}], alpha=1
//    output: z=[{3, 1}, {5, 10}, {7, 1}, {12, -1}]
// */
// void sparseAxpy(double alpha, Kokkos::View<const Element*> &x, Kokkos::View<const Element*> &y, Kokkos::View<double*> &z) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <map>
#include <unordered_map>

#include "kokkos-includes.hpp"
#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM
#include "baseline.hpp"

struct Context {
    Kokkos::View<Element*> x, y;
    Kokkos::View<double*> z;
    Kokkos::View<const Element*> x_const, y_const;
    std::vector<Element> xHost, yHost, zHost;
    size_t N;
};

void reset(Context *ctx) {
    for (size_t i = 0; i < x.size(); i++) {
        ctx->xHost[i] = {rand() % ctx->N, (rand() / (double) RAND_MAX) * 2.0 - 1.0};
        ctx->yHost[i] = {rand() % ctx->N, (rand() / (double) RAND_MAX) * 2.0 - 1.0};
    }

    std::sort(ctx->xHost.begin(), ctx->xHost.end(), [](Element const& a, Element const& b) {
        return a.index < b.index;
    });
    std::sort(ctx->yHost.begin(), ctx->yHost.end(), [](Element const& a, Element const& b) {
        return a.index < b.index;
    });

    for (size_t i = 0; i < ctx->xHost.size(); i++) {
        ctx->x(i) = ctx->xHost[i];
        ctx->y(i) = ctx->yHost[i];
    }
    ctx->x_const = ctx->x;
    ctx->y_const = ctx->y;

    std::fill(ctx->zHost.begin(), ctx->zHost.end(), 0.0);
    copyVectorToView(ctx->zHost, ctx->z);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    const size_t nVals = ctx->N * SPARSE_LA_SPARSITY;

    ctx->xHost.resize(nVals);
    ctx->yHost.resize(nVals);
    ctx->zHost.resize(ctx->N);

    ctx->x = Kokkos::View<Element*>("x", nVals);
    ctx->y = Kokkos::View<Element*>("y", nVals);
    ctx->z = Kokkos::View<double*>("z", ctx->N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    sparseAxpy(1.0, ctx->x_const, ctx->y_const, ctx->z);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctSparseAxpy(1.0, ctx->xHost, ctx->yHost, ctx->zHost);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;
    const size_t nVals = TEST_SIZE * SPARSE_LA_SPARSITY;

    Kokkos::View<Element*> x("x", nVals);
    Kokkos::View<Element*> y("y", nVals);
    Kokkos::View<double*> z("z", TEST_SIZE);

    std::vector<Element> xHost(nVals);
    std::vector<Element> yHost(nVals);
    std::vector<Element> correct(TEST_SIZE), test(TEST_SIZE);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        double alpha = (rand() / (double) RAND_MAX) * 2.0 - 1.0;
        for (size_t i = 0; i < x.size(); i++) {
            xHost[i] = {rand() % ctx->N, (rand() / (double) RAND_MAX) * 2.0 - 1.0};
        }
        std::sort(xHost.begin(), xHost.end(), [](Element const& a, Element const& b) {
            return a.index < b.index;
        });
        for (size_t i = 0; i < x.size(); i++) {
            x(i) = xHost[i];
        }
        Kokkos::View<const Element*> x_const = x;

        for (size_t i = 0; i < y.size(); i++) {
            yHost[i] = {rand() % ctx->N, (rand() / (double) RAND_MAX) * 2.0 - 1.0};
        }
        std::sort(yHost.begin(), yHost.end(), [](Element const& a, Element const& b) {
            return a.index < b.index;
        });
        for (size_t i = 0; i < y.size(); i++) {
            y(i) = yHost[i];
        }
        Kokkos::View<const Element*> y_const = y;

        std::fill(correct.begin(), correct.end(), 0.0);
        copyVectorToView(correct, z);

        // compute correct result
        correctSparseAxpy(alpha, xHost, yHost, correct);

        // compute test result
        sparseAxpy(alpha, x, y, z);

        copyViewToVector(z, test);
        
        if (!fequal(correct, test, 1e-4)) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
