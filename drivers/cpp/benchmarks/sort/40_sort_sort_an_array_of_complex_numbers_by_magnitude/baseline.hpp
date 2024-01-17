#pragma once
#include <complex>
#include <vector>

/* Sort the vector x of complex numbers by their magnitude in ascending order.
   Example:
   
   input: [3.0-1.0i, 4.5+2.1i, 0.0-1.0i, 1.0-0.0i, 0.5+0.5i]
   output: [0.5+0.5i, 0.0-1.0i, 1.0-0.0i, 3.0-1.0i, 4.5+2.1i]
*/
void NO_INLINE correctSortComplexByMagnitude(std::vector<std::complex<double>> &x) {
   std::sort(x.begin(), x.end(), [](const std::complex<double> &a, const std::complex<double> &b) {
      return std::abs(a) < std::abs(b);
   });
}