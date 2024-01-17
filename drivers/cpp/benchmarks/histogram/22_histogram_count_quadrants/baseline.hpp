#pragma once
#include <array>
#include <vector>

//struct Point {
//   double x, y;
//};

/* Count the number of cartesian points in each quadrant. The vector points contains a list of `Point` objects.
   Store the counts in `bins`.
   Example:

   input: [{x=1.5, y=0.1}, {x=-3, y=1.1}, {x=5, y=9}, {x=1.5, y=-1}, {x=3, y=-7}, {x=0.1, y=2}]
   output: [3, 1, 0, 2]
*/
void NO_INLINE correctCountQuadrants(std::vector<Point> const& points, std::array<size_t, 4> &bins) {
   for (auto const& point : points) {
      if (point.x >= 0 && point.y >= 0) {
            bins[0] += 1;
      } else if (point.x < 0 && point.y >= 0) {
            bins[1] += 1;
      } else if (point.x < 0 && point.y < 0) {
            bins[2] += 1;
      } else if (point.x >= 0 && point.y < 0) {
            bins[3] += 1;
      }
   }
}