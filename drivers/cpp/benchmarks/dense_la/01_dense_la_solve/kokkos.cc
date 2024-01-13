// Driver for 01_dense_la_solve for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* Solve the linear system Ax=b for x.
//    A is an NxN matrix. x and b have N elements.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Example:
//    
//    input: A=[[1,4,2], [1,2,3], [2,1,3]] b=[11, 11, 13]
//    output: x=[3, 1, 2]
// */
// void solveLinearSystem(Kokkos::View<const double**> &A, Kokkos::View<const double*> &b, Kokkos::View<double*> &x, size_t N) {

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

};

void reset(Context *ctx) {

}

Context *init() {
    Context *ctx = new Context();



    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {

}

void NO_OPTIMIZE best(Context *ctx) {

}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input

        // compute correct result

        // compute test result
        
        if () {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
