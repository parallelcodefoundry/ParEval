// PROMPT:
// /* Compute the prefix sum array of the vector x and return its sum.
//    Example:
// 
//    input: [-7, 2, 1, 9, 4, 8]
//    output: 15
// */
// double sumOfPrefixSum(std::vector<double> const& x) {
#pragma once

#include <numeric>
#include <vector>

double correctSumOfPrefixSum(std::vector<double> const& x) {
    std::vector<double> prefixSum(x.size());
    std::inclusive_scan(x.begin(), x.end(), prefixSum.begin());
    return std::accumulate(prefixSum.begin(), prefixSum.end(), 0.0);
}