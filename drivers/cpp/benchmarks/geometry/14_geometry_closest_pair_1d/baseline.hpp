#pragma once
#include <vector>
#include <limits>

/* Return the distance between the closest two elements in the vector x.
   Example:

   input: [7, 3, 9, 12, 31, 1]
   output: 2
*/
double NO_INLINE correctClosestPair(std::vector<double> const& x) {
    // The vector needs to have at least two elements
    if (x.size() < 2)   {
        return 0;
    }

    auto getDist = [](double const& a, double const& b) {
        return std::abs(b-a);
    };

    double minDist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < x.size() - 1; i++) {
        for (size_t j = i + 1; j < x.size(); j++) {
            const double dist = getDist(x[i], x[j]);
            if (dist < minDist) {
                minDist = dist;
            }
        }
    }

    return minDist;
}
