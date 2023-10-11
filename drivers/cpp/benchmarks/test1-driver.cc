#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


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

    std::vector<float> smallData(10000);
    for (int i = 0; i < smallData.size(); i += 1) {
        smallData[i] = rand() / (float) RAND_MAX;
    }

    const float smallSum = sum(smallData);
    const float realSum = std::reduce(ctx->data.begin(), ctx->data.end());

    return std::abs(smallSum - realSum) < 1e-6;
}

void reset(Context *ctx) {
    return;
}

void destroy(Context *ctx) {
    delete ctx;
}


