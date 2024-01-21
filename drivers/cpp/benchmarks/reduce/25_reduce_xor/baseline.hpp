#pragma once
#include <vector>
#include <numeric>

/* Return the logical XOR reduction of the vector of bools x.
   Example:

   input: [false, false, false, true]
   output: true
*/
bool correctReduceLogicalXOR(std::vector<bool> const& x) {
    return std::reduce(x.begin(), x.end(), false, [] (const auto &a, const auto &b) {
        return a != b;
    });
}
