#pragma once

#include <numeric>
#include <vector>

/* Replace the i-th element of the vector x with the minimum value from indices 0 through i.
   Examples:

   input: [8, 6, -1, 7, 3, 4, 4]
   output: [8, 6, -1, -1, -1, -1, -1]

   input: [5, 4, 6, 4, 3, 6, 1, 1]
   output: [5, 4, 4, 4, 3, 3, 1, 1]
*/
void NO_INLINE correctPartialMinimums(std::vector<float> &x) {
    std::inclusive_scan(x.begin(), x.end(), x.begin(), [] (const float &x, const float &y) {
                                                           return std::min(x, y);
                                                       },
        std::numeric_limits<float>::max());
}
