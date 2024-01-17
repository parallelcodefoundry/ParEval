#pragma once
#include <vector>

/* Return true if the vector x contains the value `target`. Return false otherwise.
   Examples:
   
   input: x=[1, 8, 2, 6, 4, 6], target=3
   output: false

   input: x=[1, 8, 2, 6, 4, 6], target=8
   output: true
*/
bool NO_INLINE correctContains(std::vector<int> const& x, int target) {
    return std::find(x.begin(), x.end(), target) != x.end();
}