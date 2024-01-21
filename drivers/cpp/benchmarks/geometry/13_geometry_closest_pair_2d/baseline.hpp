#pragma once
#include <vector>
#include <limits>

/* Return the distance between the closest two points in the vector points.
   Example:

   input: [{2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}]
   output: 1.41421
*/
double NO_INLINE correctClosestPair(std::vector<Point> const& points) {
    // The polygon needs to have at least two points
    if (points.size() < 2)   {
        return 0;
    }

    auto getDist = [](Point const& a, Point const& b) {
        return std::sqrt(std::pow(b.x-a.x, 2) + std::pow(b.y-a.y, 2));
    };

    double minDist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < points.size() - 1; i++) {
        for (size_t j = i + 1; j < points.size(); j++) {
            const double dist = getDist(points[i], points[j]);
            if (dist < minDist) {
                minDist = dist;
            }
        }
    }

    return minDist;
}
