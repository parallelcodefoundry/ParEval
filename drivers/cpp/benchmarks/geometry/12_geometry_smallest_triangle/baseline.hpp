#pragma once
#include <vector>
#include <limits>

/* Return the area of the smallest triangle that can be formed by any 3 points.
   Example:

   input: [{0, 10}, {5, 5}, {1,0}, {-1, 1}, {-10, 0}]
   output: 5.5
*/
double NO_INLINE correctSmallestArea(std::vector<Point> const& points) {
    // The polygon needs to have at least three points
    if (points.size() < 3)   {
        return 0;
    }

    auto triArea = [](Point const& a, Point const& b, Point const& c) {
        return 0.5 * std::abs((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)));
    };

    double minArea = std::numeric_limits<double>::max();
    for (size_t i = 0; i < points.size() - 2; i++) {
        for (size_t j = i + 1; j < points.size() - 1; j++) {
            for (size_t k = j + 1; k < points.size(); k++) {
                const double area = triArea(points[i], points[j], points[k]);
                if (area < minArea) {
                    minArea = area;
                }
            }
        }
    }

    return minArea;
}
