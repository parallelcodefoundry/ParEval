#pragma once

#include <algorithm>
#include <vector>


void NO_INLINE correctSortIgnoreZero(std::vector<int> &x) {
    std::vector<int> nonZeroElements;
    for (int num : x) {
        if (num != 0) {
            nonZeroElements.push_back(num);
        }
    }

    std::sort(nonZeroElements.begin(), nonZeroElements.end());

    size_t nonZeroIndex = 0;
    for (int i = 0; i < x.size(); i += 1) {
        if (x[i] != 0) {
            x[i] = nonZeroElements[nonZeroIndex];
            nonZeroIndex += 1;
        }
    }
}