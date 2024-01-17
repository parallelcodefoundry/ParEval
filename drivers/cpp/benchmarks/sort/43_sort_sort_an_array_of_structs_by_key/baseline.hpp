#pragma once
#include <vector>

// struct Result {
//    int startTime, duration;
//    float value;
// };

/* Sort vector of Result structs by start time in ascending order.
   Example:
   
   input: [{startTime=8, duration=4, value=-1.22}, {startTime=2, duration=10, value=1.0}, {startTime=10, duration=3, value=0.0}]
   output: [{startTime=2, duration=10, value=1.0}, {startTime=8, duration=4, value=-1.22}, {startTime=10, duration=3, value=0.0}]
*/
void NO_INLINE correctSortByStartTime(std::vector<Result> &results) {
   std::sort(results.begin(), results.end(), [](Result const& a, Result const& b) {
      return a.startTime < b.startTime;
   });
}