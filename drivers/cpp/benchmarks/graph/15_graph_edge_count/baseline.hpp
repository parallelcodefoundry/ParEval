#pragma once
#include <vector>
#include <queue>
#include <stack>
#include <deque>
#include <limits>

/* Count the number of edges in the directed graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major.
   Example:

	 input: [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]]
   output: 3
*/
int NO_INLINE correctEdgeCount(std::vector<int> const& A, size_t N) {
   int count = 0;
   for (int i = 0; i < N; i += 1) {
      for (int j = 0; j < N; j += 1) {
         if (A[i * N + j] == 1) {
            count += 1;
         }
      }
   }
   return count;
}