#pragma once
#include <array>
#include <vector>

/* Count the number of doubles in the vector x that have a fractional part 
   in [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1). Store the counts in `bins`.
   Examples:

   input: [7.8, 4.2, 9.1, 7.6, 0.27, 1.5, 3.8]
   output: [2, 1, 2, 2]

   input: [1.9, 0.2, 0.6, 10.1, 7.4]
   output: [2, 1, 1, 1]
*/
void NO_INLINE correctCountQuartiles(std::vector<double> const& x, std::array<size_t, 4> &bins) {
   for (int i = 0; i < x.size(); i += 1) {
      const double val = x[i];
      const double frac = val - (int) val;
      if (frac < 0.25) {
         bins[0] += 1;
      } else if (frac < 0.5) {
         bins[1] += 1;
      } else if (frac < 0.75) {
         bins[2] += 1;
      } else {
         bins[3] += 1;
      }
   }
}

#if defined(USE_CUDA)
// fix the issue where atomicAdd is not defined for size_t
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t is not 64 bits");

__device__ __forceinline__ void atomicAdd(size_t* address, size_t val) {
   atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
}
#endif