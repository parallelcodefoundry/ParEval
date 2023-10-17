// Driver for Test2. Prompt:
//  /* Use MPI to compute the average of all values across ranks and return it on each rank. Each rank has an equal sized subset of values stored in `vals`. Assume MPI has already been initialized in MPI_COMM_WORLD.
//  Example, if vals on ranks 0, 1, and 2 stores:
//      0: [1, 3, 2, 3]
//      1: [0, 0, 1, 2]
//      2: [5, 3, 9, 7]
//  The result is 3 on each rank.
//  */
//  float average(std::vector<float> const& vals) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <mpi.h>

/* generated function */
extern float average(std::vector<float> const& vals);


/* context -- data store */
struct Context {
    std::vector<float> data;
};

Context *init() {
    Context *ctx = new Context();
    ctx->data.resize(100000);
    return ctx;
}

void benchmark(Context *ctx) {
    average(ctx->data);
}

bool validate(Context *ctx) {
    
    std::vector<float> smallData(100);
    for (int i = 0; i < smallData.size(); i += 1) {
        smallData[i] = rand() / (float) RAND_MAX;
    }

    /* get correct output */
    std::vector<float> sums(100);
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Allreduce(smallData.data(), sums.data(), smallData.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    const float perElementAverage = std::reduce(sums.begin(), sums.end()) / worldSize;
    const float realAverage = perElementAverage / smallData.size();
    MPI_Barrier(MPI_COMM_WORLD);

    /* get output from generated function */
    const float generatedAverage = average(smallData);
    MPI_Barrier(MPI_COMM_WORLD);

    return std::abs(realAverage - generatedAverage) < 1e-4;
}

void reset(Context *ctx) {
    return;
}

void destroy(Context *ctx) {
    delete ctx;
}

