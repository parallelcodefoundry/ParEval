#pragma once
#include <algorithm>
#include <vector>
#include <queue>
#include <stack>
#include <deque>
#include <limits>

/* Compute the highest node degree in the undirected graph. The graph is defined in the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. A is undirected.
   Example:

	 input: [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]]
   output: 3
*/
int NO_INLINE correctMaxDegree(std::vector<int> const& A, size_t N) {
   int maxDegree = 0;
   for (int i = 0; i < N; i += 1) {
      int degree = 0;
      for (int j = 0; j < N; j += 1) {
         degree += A[i * N + j];
      }
      maxDegree = std::max(maxDegree, degree);
   }
   return maxDegree;
}