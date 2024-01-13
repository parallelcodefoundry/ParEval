#pragma once
#include <vector>
#include <array>

/* Count the number of pixels in image with each grayscale intensity.
   The vector `image` is a grayscale image with values 0-255.
   Store the results in `bins`.
   Example:
   
   input: image=[2, 116, 201, 11, 92, 92, 201, 4, 2]
   output: [0, 0, 2, 0, 1, ...]
*/
void NO_INLINE correctPixelCounts(std::vector<int> const& image, std::array<size_t, 256> &bins) {
   for (int i = 0; i < image.size(); i += 1) {
      bins[image[i]] += 1;
   }
}