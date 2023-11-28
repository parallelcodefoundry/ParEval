#pragma once

bool isPowerOfTwo(int);

/* Apply the isPowerOfTwo function to every value in x and store the results in mask.
   Example:
   
   input: [8, 0, 9, 7, 15, 64, 3]
   output: [true, false, false, false, false, true, false]
*/
void correctMapPowersOfTwo(std::vector<int> const& x, std::vector<bool> &mask) {
    for (int i = 0; i < x.size(); i++) {
        mask[i] = isPowerOfTwo(x[i]);
    }
}