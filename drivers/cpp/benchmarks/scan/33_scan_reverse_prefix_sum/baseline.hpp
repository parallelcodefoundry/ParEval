#pragma once

#include <numeric>
#include <vector>

/* Compute the reverse prefix sum of the vector x into output.
   Examples:

   input: [1, 7, 4, 6, 6, 2]
   output: [2, 8, 14, 18, 25, 26]

   input: [3, 3, 7, 1, -2]
   output: [-2, -1, 6, 9, 12]
*/
void NO_INLINE correctReversePrefixSum(std::vector<int> const& x, std::vector<int> &output) {
    std::vector<int> reverseX;
    for (int i = x.size() - 1; i >= 0; i--) {
        reverseX.push_back(x[i]);
    }
    std::inclusive_scan(reverseX.begin(), reverseX.end(), output.begin());
}
