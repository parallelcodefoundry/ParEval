#pragma once
#include <vector>

/* Return the index of the first even number in the vector x.
   Examples:

   input: [7, 3, 9, 5, 5, 7, 2, 9, 12, 11]
   output: 6

   input: [3, 8, 9, 9, 3, 4, 8, 6]
   output: 1
*/
size_t NO_INLINE correctFindFirstEven(std::vector<int> const& x) {
   for (size_t i = 0; i < x.size(); i += 1) {
      if (x[i] % 2 == 0) {
            return i;
      }
   }
   return x.size();
}