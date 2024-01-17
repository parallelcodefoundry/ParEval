#pragma once
#include <array>
#include <vector>

/* Vector x contains values between 0 and 100, inclusive. Count the number of
   values in [0,10), [10, 20), [20, 30), ... and store the counts in `bins`.
   Example:

   input: [7, 32, 95, 12, 39, 32, 11, 71, 70, 66]
   output: [1, 2, 0, 3, 0, 0, 1, 2, 0, 1]
*/
void NO_INLINE correctBinsBy10Count(std::vector<double> const& x, std::array<size_t, 10> &bins) {
   for (size_t i = 0; i < x.size(); i += 1) {
      const size_t bin = x[i] / 10;
      bins[bin] += 1;
   }
}


#if defined(USE_CUDA)
// fix the issue where atomicAdd is not defined for size_t
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t is not 64 bits");

__device__ __forceinline__ void atomicAdd(size_t* address, size_t val) {
   atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
}
#endif