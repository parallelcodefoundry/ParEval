#pragma once
#include <algorithm>
#include <numeric>
#include <vector>

/* For each value in the vector x compute its index in the sorted vector.
   Store the results in `ranks`.
   Examples:

   input: [3.1, 2.8, 9.1, 0.4, 3.14]
   output: [2, 1, 4, 0, 3]
 
   input: [100, 7.6, 16.1, 18, 7.6]
   output: [4, 0, 1, 2, 3]
*/
void NO_INLINE correctRanks(std::vector<float> const& x, std::vector<size_t> &ranks) {
   std::vector<size_t> indices(x.size());
   std::iota(indices.begin(), indices.end(), 0);

   std::sort(indices.begin(), indices.end(), [&x](size_t i1, size_t i2) {
      return x[i1] < x[i2];
   });

   for (int i = 0; i < indices.size(); i += 1) {
      ranks[indices[i]] = i;
   }
}