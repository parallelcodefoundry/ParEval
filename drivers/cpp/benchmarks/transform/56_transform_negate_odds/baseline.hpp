#pragma once

/* In the vector x negate the odd values and divide the even values by 2.
   Example:
   
   input: [16, 11, 12, 14, 1, 0, 5]
   output: [8, -11, 6, 7, -1, 0, -5]
*/
void NO_INLINE correctNegateOddsAndHalveEvens(std::vector<int> &x) {
    std::transform(x.begin(), x.end(), x.begin(), [](int i) {
        if (i % 2 == 0) {
            return i / 2;
        } else {
            return -i;
        }
    });
}