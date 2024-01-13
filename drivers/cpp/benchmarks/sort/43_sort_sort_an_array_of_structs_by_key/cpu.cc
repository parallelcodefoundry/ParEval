// Driver for 43_sort_sort_an_array_of_structs_by_key for Serial, OpenMP, MPI, and MPI+OpenMP
// struct Result {
//    int startTime, duration;
//    float value;
// };
// 
// /* Sort vector of Result structs by start time in ascending order.
//    Example:
//    
//    input: [{startTime=8, duration=4, value=-1.22}, {startTime=2, duration=10, value=1.0}, {startTime=10, duration=3, value=0.0}]
//    output: [{startTime=2, duration=10, value=1.0}, {startTime=8, duration=4, value=-1.22}, {startTime=10, duration=3, value=0.0}]
// */
// void sortByStartTime(std::vector<Result> &results) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM
#include "baseline.hpp"

struct Context {
    std::vector<Result> results;
    std::vector<int> startTime, duration;
    std::vector<float> value;
};

void reset(Context *ctx) {
    fillRand(ctx->startTime, 0, 100);
    fillRand(ctx->duration, 1, 10);
    fillRand(ctx->value, -1.0, 1.0);

    BCAST(ctx->startTime, INT);
    BCAST(ctx->duration, INT);
    BCAST(ctx->value, FLOAT);

    for (int i = 0; i < startTime.size(); i += 1) {
        ctx->results[i].startTime = ctx->startTime[i];
        ctx->results[i].duration = ctx->duration[i];
        ctx->results[i].value = ctx->value[i];
    }
}

Context *init() {
    Context *ctx = new Context();

    ctx->results.resize(DRIVER_PROBLEM_SIZE);
    ctx->startTime.resize(DRIVER_PROBLEM_SIZE);
    ctx->duration.resize(DRIVER_PROBLEM_SIZE);
    ctx->value.resize(DRIVER_PROBLEM_SIZE);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    sortByStartTime(ctx->results);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctSortByStartTime(ctx->results);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<Result> correct(TEST_SIZE), test(TEST_SIZE);
    std::vector<int> startTime(TEST_SIZE), duration(TEST_SIZE);
    std::vector<float> value(TEST_SIZE);

    int rank;
    GET_RANK(rank);

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(startTime, 0, 100);
        fillRand(duration, 1, 10);
        fillRand(value, -1.0, 1.0);

        BCAST(startTime, INT);
        BCAST(duration, INT);
        BCAST(value, FLOAT);

        for (int i = 0; i < startTime.size(); i += 1) {
            correct[i].startTime = startTime[i];
            correct[i].duration = duration[i];
            correct[i].value = value[i];

            test[i].startTime = startTime[i];
            test[i].duration = duration[i];
            test[i].value = value[i];
        }

        // compute correct result
        correctSortByStartTime(correct);

        // compute test result
        sortByStartTime(test);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && !std::equal(correct.begin(), correct.end(), test.begin())) {
            isCorrect = false;
        }
        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
