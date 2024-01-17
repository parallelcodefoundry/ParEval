#pragma once
#include <vector>
#include <numeric>

/* Return the value of the smallest odd number in the vector x.
   Examples:

   input: [7, 9, 5, 2, 8, 16, 4, 1]
   output: 1

   input: [8, 36, 7, 2, 11]
   output: 7
*/
int NO_INLINE correctSmallestOdd(std::vector<int> const& x) {
    return std::reduce(x.begin(), x.end(), std::numeric_limits<int>::max(), [] (const auto &a, const auto &b) {
        if (a < b) {
            if (a % 2 == 1) return a;
            else if (b % 2 == 1) return b;
            else return std::numeric_limits<int>::max();
        } else {
            if (b % 2 == 1) return b;
            else if (a % 2 == 1) return a;
            else return std::numeric_limits<int>::max();
        }
    });
}
