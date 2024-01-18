#pragma once
#include <vector>
#include <numeric>

/* Return the product of the vector x with every odd indexed element inverted.
   i.e. x_0 * 1/x_1 * x_2 * 1/x_3 * x_4 ...
   Example:

   input: [4, 2, 10, 4, 5]
   output: 25
*/
double NO_INLINE correctProductWithInverses(std::vector<double> const& x) {
    std::vector<double> data;
    for (size_t i = 0; i < x.size(); i++)
        data.push_back(i % 2 ? 1.0 / x[i] : x[i]);
    return std::reduce(data.begin(), data.end(), 1.0, std::multiplies());
}
