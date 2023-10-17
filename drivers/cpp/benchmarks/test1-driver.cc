#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <cstdio>


/* generated function */
extern float sum(std::vector<float> const& vals);

/* context -- data store */
struct Context {
    std::vector<float> data;
};

Context *init() {
    Context *ctx = new Context();
    ctx->data.resize(1000000);
    return ctx;
}

void benchmark(Context *ctx) {
    sum(ctx->data);
}

bool validate(Context *ctx) {

    std::vector<float> smallData(100);
    for (int i = 0; i < smallData.size(); i += 1) {
        smallData[i] = rand() / (float) RAND_MAX;
    }

    const float smallSum = sum(smallData);
    const float realSum = std::reduce(smallData.begin(), smallData.end());

    return std::abs(smallSum - realSum) < 1e-4;
}

void reset(Context *ctx) {
    return;
}

void destroy(Context *ctx) {
    delete ctx;
}


