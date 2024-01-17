#pragma once
#include <array>
#include <string>
#include <vector>

/* For each letter in the alphabet, count the number of strings in the vector s that start with that letter.
   Assume all strings are in lower case. Store the output in `bins` array.
   Example:

   input: ["dog", "cat", "xray", "cow", "code", "type", "flower"]
   output: [0, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
*/
void NO_INLINE correctFirstLetterCounts(std::vector<std::string> const& s, std::array<size_t, 26> &bins) {
   for (int i = 0; i < s.size(); i += 1) {
      const char c = s[i][0];
      const int index = c - 'a';
      bins[index] += 1;
   }
}

#if defined(USE_CUDA)
// fix the issue where atomicAdd is not defined for size_t
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t is not 64 bits");

__device__ __forceinline__ void atomicAdd(size_t* address, size_t val) {
   atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
}
#endif