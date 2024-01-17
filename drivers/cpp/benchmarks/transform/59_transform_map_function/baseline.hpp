#pragma once

bool isPowerOfTwo(int);

/* Apply the isPowerOfTwo function to every value in x and store the results in mask.
   Example:
   
   input: [8, 0, 9, 7, 15, 64, 3]
   output: [true, false, false, false, false, true, false]
*/
void NO_INLINE correctMapPowersOfTwo(std::vector<int> const& x, std::vector<bool> &mask) {
    for (int i = 0; i < x.size(); i++) {
        mask[i] = isPowerOfTwo(x[i]);
    }
}


/* THIS IS FOR THE CUDA/HIP SAMPLES WHERE CALLING THE __device__ FUNCTION WOULD BE AN ERROR ON CPU */
bool isPowerOfTwoHOST(int x) {
    return (x > 0) && !(x & (x - 1));
}

void NO_INLINE correctMapPowersOfTwoHOST(std::vector<int> const& x, std::vector<bool> &mask) {
    for (int i = 0; i < x.size(); i++) {
        mask[i] = isPowerOfTwoHOST(x[i]);
    }
}