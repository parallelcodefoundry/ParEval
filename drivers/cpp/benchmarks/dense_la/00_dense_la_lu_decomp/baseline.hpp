#pragma once
#include <vector>

/* Factorize the matrix A into A=LU where L is a lower triangular matrix and U is an upper triangular matrix.
   Store the results for L and U into the original matrix A. 
   A is an NxN matrix stored in row-major.
   Example:

   input: [[4, 3], [6, 3]]
   output: [[4, 3], [1.5, -1.5]]
*/
void correctLuFactorize(std::vector<double> &A, size_t N) {
   for (size_t k = 0; k < N; ++k) {
       for (size_t i = k + 1; i < N; ++i) {

           double factor = A[i * N + k] / A[k * N + k];
           A[i * N + k] = factor;
           
           for (size_t j = k + 1; j < N; ++j) {
               A[i * N + j] -= factor * A[k * N + j];
           }
       }
   }
}