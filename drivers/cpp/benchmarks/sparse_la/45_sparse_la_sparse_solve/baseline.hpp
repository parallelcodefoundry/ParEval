#pragma once
#include <vector>

// struct COOElement {
//    size_t row, column;
//    double value;
// };

/* Solve the sparse linear system Ax=b for x.
   A is a sparse NxN matrix in COO format. x and b are dense vectors with N elements.
   Example:
   
   input: A=[{0,0,1}, {0,1,1}, {1,1,-2}] b=[1,4]
   output: x=[3,-2]
*/
void NO_INLINE correctSolveLinearSystem(std::vector<COOElement> const& A, std::vector<double> const& b, std::vector<double> &x, size_t N) {
   std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));   
   std::vector<double> b_copy = b;

   // Fill the matrix with the values from A
   for (const auto& element : A) {
      matrix[element.row][element.column] = element.value;
   }

   // Initialize x with the size N
   x.assign(N, 0.0);

   // Perform Gaussian elimination
   for (size_t i = 0; i < N; ++i) {
      // Find pivot
      double maxEl = std::abs(matrix[i][i]);
      size_t maxRow = i;
      for (size_t k = i + 1; k < N; ++k) {
         if (std::abs(matrix[k][i]) > maxEl) {
               maxEl = std::abs(matrix[k][i]);
               maxRow = k;
         }
      }

      // Swap maximum row with current row (column by column)
      for (size_t k = i; k < N; ++k) {
         std::swap(matrix[maxRow][k], matrix[i][k]);
      }
      std::swap(b_copy[maxRow], b_copy[i]);

      // Make all rows below this one 0 in the current column
      for (size_t k = i + 1; k < N; ++k) {
         double c = -matrix[k][i] / matrix[i][i];
         for (size_t j = i; j < N; ++j) {
               if (i == j) {
                  matrix[k][j] = 0;
               } else {
                  matrix[k][j] += c * matrix[i][j];
               }
         }
         b_copy[k] += c * b_copy[i];
      }
   }

   // Solve equation Ax=b for an upper triangular matrix A
   for (int i = N - 1; i >= 0; --i) {
      x[i] = b_copy[i] / matrix[i][i];
      for (int k = i - 1; k >= 0; --k) {
         b_copy[k] -= matrix[k][i] * x[i];
      }
   }
}