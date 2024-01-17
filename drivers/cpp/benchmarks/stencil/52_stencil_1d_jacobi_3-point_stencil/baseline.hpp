#pragma once
#include <vector>

/* Compute one iteration of a 3-point 1D jacobi stencil on `input`. Store the results in `output`.
   Each element of `input` will be averaged with its two neighbors and stored in the corresponding element of `output`.
   i.e. output[i] = (input[i-1]+input[i]+input[i+1])/3
   Replace with 0 when reading past the boundaries of `input`.
   Example:

   input: [9, -6, -1, 2, 3]
   output: [1, 2/3, -5/3, 4/3, 5/3]
*/
void NO_INLINE correctJacobi1D(std::vector<double> const& input, std::vector<double> &output) {
    for (size_t i = 0; i < input.size(); i++) {
        double sum = 0.0;
        if (i > 0) {
            sum += input[i - 1];
        }
        if (i < input.size() - 1) {
            sum += input[i + 1];
        }
        sum += input[i];
        output[i] = sum / 3.0;
    }
}
