#pragma once
#include <vector>

/* Replace every element of the vector x with 1-1/x.
   Example:

   input: [2, 4, 1, 12, -2]
   output: [0.5, 0.75, 0, 0.91666666, 1.5]
*/
void NO_INLINE correctOneMinusInverse(std::vector<double> &x) {
    std::transform(x.begin(), x.end(), x.begin(), [](double x) { return 1.0 - 1.0 / x; });
}