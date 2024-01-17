#pragma once
#include <vector>

/* Return the index of the value in the vector x that is closest to the math constant PI.
   Use M_PI for the value of PI.
   Example:

   input: [9.18, 3.05, 7.24, 11.3, -166.49, 2.1]
   output: 1
*/
size_t NO_INLINE correctFindClosestToPi(std::vector<double> const& x) {
   size_t index = 0;
   double min = std::abs(x[0] - M_PI);
   for (size_t i = 1; i < x.size(); ++i) {
      double diff = std::abs(x[i] - M_PI);
      if (diff < min) {
            min = diff;
            index = i;
      }
   }
   return index;
}