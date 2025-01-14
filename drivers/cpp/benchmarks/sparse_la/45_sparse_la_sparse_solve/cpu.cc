// Driver for 45_sparse_la_sparse_solve for Serial, OpenMP, MPI, and MPI+OpenMP
// struct COOElement {
//    size_t row, column;
//    double value;
// };
// 
// /* Solve the sparse linear system Ax=b for x.
//    A is a sparse NxN matrix in COO format. x and b are dense vectors with N elements.
//    Example:
//    
//    input: A=[{0,0,1}, {0,1,1}, {1,1,-2}] b=[1,4]
//    output: x=[3,-2]
// */
// void solveLinearSystem(std::vector<COOElement> const& A, std::vector<double> const& b, std::vector<double> &x, size_t N) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM
#include "baseline.hpp"

struct Context {
    std::vector<size_t> A_rows, A_columns;
    std::vector<double> A_values;

    std::vector<COOElement> A;
    std::vector<double> b, x;
    size_t N;
};

void sortCOOElements(std::vector<COOElement> &vec) {
    std::sort(vec.begin(), vec.end(), [](const COOElement &a, const COOElement &b) {
        return (a.row == b.row) ? (a.column < b.column) : (a.row < b.row);
    });
}

void createRandomLinearSystem(std::vector<COOElement> &A, std::vector<size_t> &A_rows, std::vector<size_t> &A_columns, 
    std::vector<double> &A_values, std::vector<double> &b, std::vector<double> &x, size_t N) {
    
    fillRand(A_rows, 0UL, N);
    fillRand(A_columns, 0UL, N);
    fillRand(A_values, -10.0, 10.0);
    BCAST(A_rows, UNSIGNED_LONG);
    BCAST(A_columns, UNSIGNED_LONG);
    BCAST(A_values, DOUBLE);

    for (int i = 0; i < A_rows.size(); i += 1) {
        A[i] = {A_rows[i], A_columns[i], A_values[i]};
    }
    sortCOOElements(A);

    fillRand(x, -10.0, 10.0);

    std::fill(b.begin(), b.end(), 0.0);
    for (size_t i = 0; i < A.size(); i += 1) {
        b[A[i].row] += A[i].value * x[A[i].column];
    }
    BCAST(b, DOUBLE);

    std::fill(x.begin(), x.end(), 0.0); // every rank resets x
}


void reset(Context *ctx) {
    createRandomLinearSystem(ctx->A, ctx->A_rows, ctx->A_columns, ctx->A_values, ctx->b, ctx->x, ctx->N);
}

Context *init() {
    Context *ctx = new Context();

    ctx->N = DRIVER_PROBLEM_SIZE;
    const size_t nVals = ctx->N * ctx->N * SPARSE_LA_SPARSITY;

    ctx->A_rows.resize(nVals);
    ctx->A_columns.resize(nVals);
    ctx->A_values.resize(nVals);
    ctx->A.resize(nVals);

    ctx->b.resize(ctx->N);
    ctx->x.resize(ctx->N);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    solveLinearSystem(ctx->A, ctx->b, ctx->x, ctx->N);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctSolveLinearSystem(ctx->A, ctx->b, ctx->x, ctx->N);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 128;
    const size_t nVals = TEST_SIZE * TEST_SIZE * SPARSE_LA_SPARSITY;

    std::vector<size_t> A_rows(nVals), A_columns(nVals);
    std::vector<double> A_values(nVals), b(TEST_SIZE), x_correct(TEST_SIZE), x_test(TEST_SIZE);
    std::vector<COOElement> A(nVals);

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        createRandomLinearSystem(A, A_rows, A_columns, A_values, b, x_correct, TEST_SIZE);
        std::fill(x_test.begin(), x_test.end(), 0.0);

        // compute correct result
        correctSolveLinearSystem(A, b, x_correct, TEST_SIZE);

        // compute test result
        solveLinearSystem(A, b, x_test, TEST_SIZE);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && !fequal(x_correct, x_test, 1e-3)) {
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
