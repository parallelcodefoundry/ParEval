#pragma once
#include <vector>

// defined in prompt
// struct COOElement {
//    size_t row, column;
//    double value;
// };

/* Factorize the sparse matrix A into A=LU where L is a lower triangular matrix and U is an upper triangular matrix.
   A is a sparse NxN matrix stored in COO format. L and U are NxN matrices in row-major.
   Example:

   input: A=[{0,0,4}, {0,1,3}, {1,0,6}, {1,1,3}]
   output: L=[{0,0,1},{1,0,1.5}, {1,1,1}] U=[{0,0,4}, {0,1,3}, {1,1,-1.5}]
*/
void NO_INLINE correctLuFactorize(std::vector<COOElement> const& A, std::vector<double> &L, std::vector<double> &U, size_t N) {
   std::vector<std::vector<double>> fullA(N, std::vector<double>(N, 0));
   for (const auto& element : A) {
      fullA[element.row][element.column] = element.value;
   }
   
   // LU factorization algorithm
   for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
         if (j >= i) {
               U[i * N + j] = fullA[i][j];
               for (size_t k = 0; k < i; ++k) {
                  U[i * N + j] -= L[i * N + k] * U[k * N + j];
               }
         }
         if (i > j) {
               L[i * N + j] = fullA[i][j] / U[j * N + j];
               for (size_t k = 0; k < j; ++k) {
                  L[i * N + j] -= L[i * N + k] * U[k * N + j] / U[j * N + j];
               }
         }
      }
      L[i * N + i] = 1;
   }
}