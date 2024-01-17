#pragma once
#include <vector>
#include <queue>
#include <stack>
#include <deque>
#include <limits>

void dfs(std::vector<int> const& A, int node, size_t N, std::vector<bool> &visited, int &count) {
   visited[node] = true;
   count += 1;
   for (int i = 0; i < N; i += 1) {
      if (A[node * N + i] == 1 && !visited[i]) {
         dfs(A, i, N, visited, count);
      }
   }
}

/* Return the number of vertices in the largest component of the graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major.
   Example:

	 input: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
   output: 2
*/
int NO_INLINE correctLargestComponent(std::vector<int> const& A, size_t N) {
   std::vector<bool> visited(N, false);
   int maxCount = 0;
   for (int i = 0; i < N; i += 1) {
      if (!visited[i]) {
         int count = 0;
         dfs(A, i, N, visited, count);
         maxCount = std::max(maxCount, count);
      }
   }
   return maxCount;
}