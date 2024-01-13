#pragma once
#include <vector>
#include <complex>
#include <cmath>

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

/* Compute the discrete fourier transform of x. Store the result in output.
   Example:

   input: [1, 4, 9, 16]
   output: [30+0i, -8-12i, -10-0i, -8+12i]
*/
void NO_INLINE correctDft(std::vector<double> const& x, std::vector<std::complex<double>> &output) {
   int N = x.size();
   output.resize(N, std::complex<double>(0, 0)); // Resize the output vector and initialize with 0

   for (int k = 0; k < N; k++) { // For each output element
      std::complex<double> sum(0, 0);
      for (int n = 0; n < N; n++) { // For each input element
         double angle = 2 * M_PI * n * k / N;
         std::complex<double> c(std::cos(angle), -std::sin(angle)); // Euler's formula
         sum += x[n] * c;
      }
      output[k] = sum;
   }
}