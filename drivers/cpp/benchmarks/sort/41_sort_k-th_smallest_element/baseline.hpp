#pragma once
#include <algorithm>
#include <vector>

/* Find the k-th smallest element of the vector x.
   Example:
   
   input: x=[1, 7, 6, 0, 2, 2, 10, 6], k=4
   output: 6
*/
int NO_INLINE correctFindKthSmallest(std::vector<int> const& x, int k) {
   std::vector<int> x_copy = x;
   std::sort(x_copy.begin(), x_copy.end());
   return x_copy[k-1];
}