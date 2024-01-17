#pragma once
#include <vector>
#include <queue>
#include <stack>
#include <deque>
#include <limits>

void dfs(std::vector<int> const& A, int node, size_t N, std::vector<bool> &visited) {
   visited[node] = true;
   for (int i = 0; i < N; i += 1) {
      if (A[node * N + i] == 1 && !visited[i]) {
         dfs(A, i, N, visited);
      }
   }
}

/* Count the number of connected components in the undirected graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. A is an undirected graph.
   Example:

	 input: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
   output: 2
*/
int NO_INLINE correctComponentCount(std::vector<int> const& A, size_t N) {
   std::vector<bool> visited(N, false);
   int count = 0;
   for (int i = 0; i < N; i += 1) {
      if (!visited[i]) {
         dfs(A, i, N, visited);
         count += 1;
      }
   }
   return count;
}