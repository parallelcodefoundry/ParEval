#pragma once
#include <vector>
#include <limits>

/* Return the largest sum of any contiguous subarray in the vector x.
   i.e. if x=[−2, 1, −3, 4, −1, 2, 1, −5, 4] then [4, −1, 2, 1] is the contiguous
   subarray with the largest sum of 6.
   Example:

   input: [−2, 1, −3, 4, −1, 2, 1, −5, 4]
   output: 6
*/
int NO_INLINE correctMaximumSubarray(std::vector<int> const& x) {
    int largestSum = std::numeric_limits<int>::lowest();
    int currSum = 0;
    for (int i = 0; i < x.size(); i++) {
        for (int j = i; j < x.size(); j++) {
            currSum += x[j];
            if (currSum > largestSum) largestSum = currSum;
        }
    }
    return largestSum;
}
