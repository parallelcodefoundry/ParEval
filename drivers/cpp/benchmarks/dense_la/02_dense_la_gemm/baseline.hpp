#pragma once
#include <vector>

/* Multiply the matrix A by the matrix B. Store the results in the matrix C.
   A is an MxK matrix, B is a KxN matrix, and C is a MxN matrix. The matrices are stored in row-major.
   Example:

   input: A=[[1, -1, 2], [0, -2, 1]] B=[[4, 1], [-1, 0], [2, 2]]
   output: C=[[9, 5], [4, 2]]
*/
void NO_INLINE correctGemm(std::vector<double> const& A, std::vector<double> const& B, std::vector<double> &C, size_t M, size_t K, size_t N) {
   for (size_t i = 0; i < M; i += 1) {
      for (size_t k = 0; k < K; k += 1) {
         for (size_t j = 0; j < N; j += 1) {
            C[i*N + j] += A[i*K + k] * B[k*N + j];
         }
      }
   }
}