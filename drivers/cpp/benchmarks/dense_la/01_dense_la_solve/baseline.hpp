#pragma once
#include <vector>

/* Solve the linear system Ax=b for x.
   A is an NxN matrix in row-major. x and b have N elements.
   Example:
   
   input: A=[[1,4,2], [1,2,3], [2,1,3]] b=[11, 11, 13]
   output: x=[3, 1, 2]
*/
void NO_INLINE correctSolveLinearSystem(std::vector<double> const& A, std::vector<double> const& b, std::vector<double> &x, size_t N) {
   // Create a copy of A to perform Gaussian elimination
   std::vector<double> A_copy = A;
   std::vector<double> b_copy = b;

   // Gaussian elimination
   for (size_t i = 0; i < N - 1; i++) {
      // Find the pivot element
      double pivot = A_copy[i * N + i];

      // Check if the pivot is zero
      if (pivot == 0) {
         return;
      }

      // Eliminate the elements below the pivot
      for (size_t j = i + 1; j < N; j++) {
         double factor = A_copy[j * N + i] / pivot;
         for (size_t k = i; k < N; k++) {
            A_copy[j * N + k] -= factor * A_copy[i * N + k];
         }
         b_copy[j] -= factor * b_copy[i];
      }
   }

   // Back substitution
   for (int i = N - 1; i >= 0; i--) {
      double sum = 0;
      for (size_t j = i + 1; j < N; j++) {
         sum += A_copy[i * N + j] * x[j];
      }
      x[i] = (b_copy[i] - sum) / A_copy[i * N + i];
   }
}