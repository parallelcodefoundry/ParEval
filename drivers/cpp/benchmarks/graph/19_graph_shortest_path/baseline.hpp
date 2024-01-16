#pragma once
#include <vector>
#include <queue>
#include <stack>
#include <deque>
#include <limits>
#include <climits>

/* Return the length of the shortest path from source to dest in the undirected graph defined by the adjacency matrix A.
   A is an NxN adjacency matrix stored in row-major. Assume the graph is connected. A is undirected.
   Example:

	 input: [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], source=0, dest=3
   output: 2
*/
int NO_INLINE correctShortestPathLength(std::vector<int> const& A, size_t N, int source, int dest) {
   std::vector<bool> visited(N, false);
   std::queue<std::pair<int, int>> queue;
   visited[source] = true;
   queue.push({source, 0});

   while (!queue.empty()) {
      auto [current, pathLength] = queue.front();
      queue.pop();

      if (current == dest) {
         return pathLength;
      }

      // Check all adjacent vertices
      for (int i = 0; i < N; ++i) {
         if (A[current*N + i] && !visited[i]) {
               visited[i] = true;
               queue.push({i, pathLength + 1});
         }
      }
   }

   return std::numeric_limits<int>::max();
}