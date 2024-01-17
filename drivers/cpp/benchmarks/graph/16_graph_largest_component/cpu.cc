// Driver for 16_graph_largest_component for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Return the number of vertices in the largest component of the undirected graph defined by the adjacency matrix A.
//    A is an NxN adjacency matrix stored in row-major. A is an undirected graph.
//    Example:
// 
// 	 input: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
//    output: 2
// */
// int largestComponent(std::vector<int> const& A, size_t N) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    std::vector<int> A;
    size_t N;
};

void fillRandomUndirectedGraph(std::vector<int> &A, size_t N) {
    std::fill(A.begin(), A.end(), 0);
    for (int i = 0; i < N; i += 1) {
        A[i * N + i] = 0;
        for (int j = i + 1; j < N; j += 1) {
            A[i * N + j] = rand() % 2;
            A[j * N + i] = A[i * N + j];
        }
    }
}

void reset(Context *ctx) {
    fillRandomUndirectedGraph(ctx->A, ctx->N);
    BCAST(ctx->A, INT);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    ctx->A.resize(ctx->N * ctx->N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    int lc = largestComponent(ctx->A, ctx->N);
    (void)lc;
}

void NO_OPTIMIZE best(Context *ctx) {
    int lc = correctLargestComponent(ctx->A, ctx->N);
    (void)lc;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 128;

    std::vector<int> A(TEST_SIZE * TEST_SIZE);

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRandomUndirectedGraph(A, TEST_SIZE);
        BCAST(A, INT);

        // compute correct result
        int correct = correctLargestComponent(A, TEST_SIZE);

        // compute test result
        int test = largestComponent(A, TEST_SIZE);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && correct != test) {
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