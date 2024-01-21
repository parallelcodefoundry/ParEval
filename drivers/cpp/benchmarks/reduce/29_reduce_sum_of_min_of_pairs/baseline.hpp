#pragma once
#include <vector>
#include <algorithm>
#include <numeric>

/* Return the sum of the minimum value at each index of vectors x and y for all indices.
   i.e. sum = min(x_0, y_0) + min(x_1, y_1) + min(x_2, y_2) + ...
   Example:

   input: x=[3, 4, 0, 2, 3], y=[2, 5, 3, 1, 7]
   output: 10
*/
double NO_INLINE correctSumOfMinimumElements(std::vector<double> const& x, std::vector<double> const& y) {
    std::vector<double> z;
    z.resize(x.size());
    std::transform(x.begin(), x.end(), y.begin(), z.begin(), [] (const auto &a, const auto &b) {
        return std::min(a, b);
    });
    return std::reduce(z.begin(), z.end());
}
