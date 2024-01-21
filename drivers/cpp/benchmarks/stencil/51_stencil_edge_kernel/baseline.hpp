#pragma once
#include <vector>

// const int edgeKernel[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};

/* Convolve the edge kernel with a grayscale image. Each pixel will be replaced with
   the dot product of itself and its neighbors with the edge kernel.
   Use a value of 0 for pixels outside the image's boundaries and clip outputs between 0 and 255.
   imageIn and imageOut are NxN grayscale images stored in row-major.
   Store the output of the computation in imageOut.
   Example:

   input: [[112, 118, 141, 152],
           [93, 101, 119, 203],
           [45, 17, 16, 232],
           [82, 31, 49, 101]]
   output: [[255, 255, 255, 255],
            [255, 147, 0, 255],
            [36, 0, 0, 255],
            [255, 39, 0, 255]]
*/
void NO_INLINE correctConvolveKernel(std::vector<int> const& imageIn, std::vector<int> &imageOut, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++){
            int sum = 0;
            for (int k = -1; k < 2; k++) {
                for (int l = -1; l < 2; l++){
                    int x = i + k;
                    int y = j + l;
                    if ((x < 0) || (x >= N) || (y < 0) || (y >= N)) {
                        sum += 0;
                    } else {
                        sum += imageIn[x * N + y] * edgeKernel[k + 1][l + 1];
                    }
                }
            }
            if (sum < 0) {
                imageOut[i * N + j] = 0;
            } else if (sum > 255) {
                imageOut[i * N + j] = 255;
            } else {
                imageOut[i * N + j] = sum;
            }
        }
    }
}
