#pragma once
#include <vector>

// struct Element {
//     size_t index;
//     double value;
// };

/* Compute z = alpha*x+y where x and y are sparse vectors. Store the result in z.
   Example:
   
   input: x=[{5, 12}, {8, 3}, {12, -1}], y=[{3, 1}, {5, -2}, {7, 1}, {8, -3}], alpha=1
   output: z=[{3, 1}, {5, 10}, {7, 1}, {12, -1}]
*/
void NO_INLINE correctSparseAxpy(double alpha, std::vector<Element> const& x, std::vector<Element> const& y, std::vector<double> &z) {
    size_t xi = 0, yi = 0;

    while (xi < x.size() && yi < y.size()) {
        if (x[xi].index < y[yi].index) {
            z[x[xi].index] += alpha * x[xi].value;
            ++xi;
        } else if (x[xi].index > y[yi].index) {
            z[y[yi].index] += y[yi].value;
            ++yi;
        } else {
            z[x[xi].index] += alpha * x[xi].value + y[yi].value;
            ++xi;
            ++yi;
        }
    }

    while (xi < x.size()) {
        z[x[xi].index] += alpha * x[xi].value;
        ++xi;
    }

    while (yi < y.size()) {
        z[y[yi].index] += y[yi].value;
        ++yi;
    }
}